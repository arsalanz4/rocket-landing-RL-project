"""
2D Rocket Landing Environment with curriculum support.

All stages use a 2D action space [throttle, gimbal_delta] from the start.
Gimbal = clip(PD(angle, ang_vel) * pd_gain + agent_action, -1, 1)
The PD inner loop keeps the rocket upright; the agent's action is a steering
correction on top of it. pd_gain decreases across stages so the agent takes
progressively more attitude authority.

Stages
------
  1  pad=40m  alt=150m  vy=-5   x=+/-10m   pd_gain=1.0  (learn to fire)
  2  pad=30m  alt=200m  vy=-10  x=+/-25m   pd_gain=1.0  (meaningful steering)
  3  pad=20m  alt=250m  vy=-10  x=+/-40m   pd_gain=1.0  (pad < spawn, must divert)
  4  pad=20m  alt=350m  vy=-15  x=+/-60m   pd_gain=0.7  (random vx + tilt at spawn)
  5  pad=20m  alt=500m  vy=-20  x=+/-80m   pd_gain=0.5  (full divert problem)
  6  pad=20m  alt=500m  vy=-20  x=+/-80m   pd_gain=0.3  (agent owns attitude)

Reward
------
  Potential-based shaping (PBRS): r = gamma * Phi(s') - Phi(s)
  Phi encodes: weighted distance to pad, speed-profile error, tilt.
  PBRS is policy-invariant -- it cannot shift the optimal policy, only accelerate
  learning. Terminals alone define success/failure objectives.

Coordinate system
-----------------
  x     : horizontal position (m), positive = right
  y     : altitude (m), positive = up
  vx    : horizontal velocity (m/s)
  vy    : vertical velocity (m/s), negative = falling
  angle : body tilt from vertical (rad), positive = tilted right
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---- Physical constants ------------------------------------------------------
GRAVITY           = 9.81
ROCKET_MASS       = 50.0
FUEL_CAPACITY     = 60.0
MAX_FUEL          = 120.0   # obs-space upper bound; stages 9-12 can carry more than 60 kg
MAX_WIND          = 20.0    # obs-space upper bound for wind_x (m/s²)
ENGINE_THRUST     = 1200.0
FUEL_BURN_RATE    = 2.0

MAX_GIMBAL_ANGLE  = np.radians(20)
# 30 deg/s slew limit -> at DT=0.05: 1.5 deg/step -> 1.5/20 = 0.075 normalized
MAX_GIMBAL_SLEW   = 0.075
MOMENT_OF_INERTIA = 800.0
GIMBAL_ARM        = 0.5
ANGULAR_DAMPING   = 0.94
MAX_TILT          = np.radians(60)

MAX_LANDING_VY    = 5.0
# Reverted 8.0->3.0 (original value). The 8.0 widen was tried after stage 7
# plateaued for ~18.8M steps, but training ~2.7-3.6M steps forward from the
# same checkpoint under the widened threshold reproducibly drifted into a
# tight one-directional vx lock twice in a row (vx clustered -33 to -43,
# unusually high leftover fuel 2-28kg vs the normal 0.0kg) -- reproducible
# across two independent rollback-and-retrain attempts from the same step-135M
# checkpoint, which rules out coincidental PPO drift as the explanation even
# though the exact reward-math mechanism causing it isn't fully understood
# (the change only altered the terminal reward for vx in [3,8], not vx>8
# where the lock sits -- likely a second-order effect on the value function's
# learned landscape rather than a direct one-step reward change). Reverting
# to the known-stable 3.0 rather than continuing to retrain into the same
# failure a third time.
MAX_LANDING_VX    = 3.0
DT                = 0.05
MAX_STEPS         = 2000

# ---- Curriculum stages -------------------------------------------------------
# pd_gain: weight of the PD stabiliser in the residual gimbal formula.
# vx_range / angle_range: random initial horizontal velocity and tilt (stages 4+).
STAGES = {
    # Stage 0: diagnostic vx-isolation pretraining, inserted ahead of stage 1.
    # Not part of the normal progression -- used once to test whether the
    # current (already-trained) policy can learn to track vx_desired at all
    # when vy is trivially easy (alt=50, vy0=-3, full pd_gain=1.0 attitude
    # authority) and x_range/vx_range give it a real horizontal task. Reward
    # function is untouched -- vy bonus terms stay fully active, they're just
    # easy to satisfy here. If the existing policy still can't pass this
    # (existing landing check already requires |vx|<=MAX_LANDING_VX=3.0 at
    # touchdown), that rules out "vy signal competition" as the cause of the
    # vx direction-lock seen at stage 9, since there's essentially no
    # competition here.
    0: dict(pad=20.0, alt=50.0, vy=-3.0, x_range=50.0, vx_range=10.0, angle_range=0.05, pd_gain=1.0, fuel=72.0),
    1: dict(pad=40.0, alt=150.0, vy=-5.0,  x_range=10.0, vx_range=0.0, angle_range=0.05, pd_gain=1.0),
    2: dict(pad=30.0, alt=200.0, vy=-10.0, x_range=25.0, vx_range=0.0, angle_range=0.05, pd_gain=1.0),
    3: dict(pad=20.0, alt=250.0, vy=-10.0, x_range=40.0, vx_range=0.0, angle_range=0.05, pd_gain=1.0),
    # Stage 4: increase altitude and speed only — no random vx/angle yet, keep pd_gain high
    4: dict(pad=20.0, alt=350.0, vy=-15.0, x_range=60.0, vx_range=0.0, angle_range=0.05, pd_gain=1.0),
    # Stages 5-7: 500m altitude requires more fuel — 20% extra (72 kg vs 60)
    # Stage 5 vy reduced 20->15 (matching stage 4): stage 4->5 was stacking THREE
    # new challenges simultaneously (pd_gain 1.0->0.7, vx_range 0->5,
    # angle_range 0.05->0.17) on top of a harder vy target at fuel=72's marginal
    # TWR=1.0027 -- same TWR a prior model lineage did successfully learn to
    # brake at (per the stage 8 comment below), so this isn't a structural
    # infeasibility like the original stage-9 fuel/TWR bug, but the current
    # model (freshly re-acquired via the stage-0 diagnostic detour + rapid
    # 1-4 sprint) showed a genuine stall here: 7.8M steps with zero variance
    # in vy/vx/reward, unlike every other stage transition this session which
    # showed at least partial drift within 1-2M steps. Isolating the new
    # pd_gain/vx_range/angle_range elements from a simultaneously harder vy
    # target should let it re-acquire the skill without three things at once.
    # Fuel dropped back to the default 60kg (was 72kg): the extra 12kg was
    # justified by vy=-20's higher braking demand, but vy is now -15 here
    # (matching stage 4, which already uses 60kg successfully) so that
    # rationale no longer applies. After the vy fix above solved vertical
    # braking, vx exploded into a much larger runaway (magnitude ~100-120,
    # up from the pre-fix lock's ~60, plateaued across 3 consecutive checks
    # spanning ~6M steps with 50% of episodes burning 100% of fuel on one
    # huge horizontal push) -- the extra fuel that vy no longer needed was
    # apparently being spent entirely on an uncorrected horizontal burn.
    # 60kg gives TWR=1.112 (up from 1.0027 at 72kg) -- strictly better
    # vertical braking margin -- while directly capping the total impulse
    # available for a runaway horizontal burn, regardless of what the
    # reward function does or doesn't penalize.
    5: dict(pad=20.0, alt=500.0, vy=-15.0, x_range=80.0, vx_range=5.0, angle_range=0.17, pd_gain=0.7, fuel=60.0),
    # vy reduced 20->15 (matching stage 5's fix): stage 6 hit the exact same
    # TWR=1.0027 stall/runaway pattern as stage 5 after ~2M steps of healthy
    # progress (success 20%->0%, ep_rew_mean flipped positive->steeply negative,
    # vx runaway to 40-80+ magnitude in 8/10 eval episodes while vy stayed
    # well-controlled). Applying the same de-stack fix first; if vx runs away
    # worse once vy braking gets cheaper (as happened at stage 5), fuel will
    # need to drop too (currently still 72kg, unchanged from before).
    6: dict(pad=20.0, alt=500.0, vy=-15.0, x_range=80.0, vx_range=5.0, angle_range=0.17, pd_gain=0.5, fuel=72.0),
    # Stage 7 fuel restored to 72 kg (matches stages 5-6): at 94 kg the rocket has
    # TWR=0.85 — full throttle still accelerates it downward for the first 11 s,
    # making the braking physics qualitatively different from stages 5-6 and causing
    # the transferred policy to give up on throttling.  At 72 kg TWR=1.002, same
    # marginal-hover regime the stage-6 policy was trained on.
    # vy reduced 20->15 (same fix as stages 5-6): stage 6 itself now trains at
    # vy=-15 (its own fix above), so stage 7 was re-imposing the harder -20
    # target the transferred policy no longer practiced, on the same marginal
    # TWR=1.0027 budget. Result matched the stages 5-6 signature exactly: vy
    # stayed well-controlled but vx developed a one-directional bias (7/10 eval
    # episodes positive, 5 in the +35-48 magnitude range), success oscillating
    # 0-35% (mostly 5-20%) for ~11.9M cumulative steps with no upward trend --
    # longer than stage 5's stall took to manifest, so not just slow transition
    # turbulence.
    # fuel reduced 72->60kg (same second-step fix as stage 5): the vy fix alone
    # made the vx bias WORSE and more consistent -- 9/10 eval episodes locked to
    # a tight +26 to +42.5 positive cluster (up from 7/10 with a looser, partly
    # negative spread pre-fix), success dropped to 0/10. Same mechanism as
    # stage 5: cheaper vy braking freed fuel headroom the policy spent on an
    # uncorrected horizontal burn. 60kg improves TWR 1.0027->1.112 and caps the
    # impulse available for the runaway burn.
    # x_range reverted 60->80 (back to original). The 60 reduction was tried
    # alongside a resume-time learning-rate cut as a low-risk pass at the
    # ~18.8M-step plateau, but the same tight one-directional vx lock that hit
    # the MAX_LANDING_VX widen experiment reappeared a third time under this
    # completely different, reward-untouched change -- vx clustered -38 to -42
    # again, abnormal leftover fuel again. Three independent failures from the
    # same step-135M checkpoint region, under three different interventions,
    # points at the checkpoint/training-maturity itself being fragile rather
    # than any specific change being the trigger. Reverting to fully stock
    # (x_range=80, and train_ppo.py's RESUME_LEARNING_RATE back to 3e-4) to
    # run a clean control test: does this checkpoint lock even with ZERO
    # changes, given enough steps? See RESUME_LEARNING_RATE comment in
    # train_ppo.py for the paired revert.
    #
    # Result: ran a clean control test AND a near-touchdown |vx| dense-bonus
    # reward change (both reward-untouched-vs-reward-shaping variants) --
    # neither moved the needle. Control test reproduced the same lock a 3rd
    # time; the dense bonus ran 7.5M steps (135.02M-142.54M) and averaged
    # 4.4% success with no upward trend (4.8% first half, 4.0% second half),
    # so it was reverted too. Both reward-adjacent and reward-neutral retrains
    # of this exact checkpoint region converge to the same bimodal ~8-12%
    # plateau (near-perfect landing OR a 20-49 m/s miss, little in between).
    # Confirmed-healthy baseline for this plateau is the checkpoint at step
    # 135,000,000 (paused_stage7_20260805_171933): 8/50 (seed 42, fixed-seed
    # 50-episode eval), pos/neg misses roughly balanced (21/25) -- NOT
    # direction-locked, matching the described bimodal shape.
    #
    # Stage 7 (new): vx-isolation pretraining at pd_gain=0.3, the exact
    # attitude-authority level the plateau above trains under. Stage 0
    # already showed the vx direction-lock is learnable, not a true local
    # optimum -- but only at pd_gain=1.0 (full PD authority), leaving the
    # pd_gain=0.3 actuator-contention hypothesis (gimbal doing double duty
    # for both attitude AND horizontal steering -- see stage 9's comment)
    # untested in isolation from vy. Same alt/x_range/vx_range/angle_range/
    # pd_gain/fuel as stage 8 below, so the horizontal-control task and the
    # 500m "must establish trim early" dynamic are identical -- only vy is
    # trivialised (-3 instead of -15), removing the vy braking/fuel-TWR
    # contention already isolated and fixed at earlier stages. If the policy
    # learns to track vx_desired here, pd_gain=0.3 alone isn't the blocker
    # and the plateau lives elsewhere (e.g. genuinely needs the near-
    # touchdown reward-shaping direction, despite the first attempt above
    # failing); if it can't, that isolates actuator-authority contention as
    # the real cause, independent of vy entirely -- point for further
    # reward/curriculum work either way.
    7: dict(pad=20.0, alt=500.0, vy=-3.0, x_range=80.0, vx_range=5.0, angle_range=0.17, pd_gain=0.3, fuel=60.0),
    8: dict(pad=20.0, alt=500.0,  vy=-15.0, x_range=80.0,  vx_range=5.0, angle_range=0.17, pd_gain=0.3, fuel=60.0),
    # Stage 9 (was "8a"): transitional stage isolating the pd_gain attitude-
    # authority handoff from the wind challenge. vy=-10 (down from -15, itself
    # down from the original -20) -- less kinetic energy to shed, so it's
    # solvable even at the ~TWR=1.0 marginal-hover regime. Fuel stays 72kg:
    # raising it lowers TWR further (ENGINE_THRUST is fixed at 1200N across
    # all stages, so more fuel = more mass = strictly less thrust-to-weight,
    # never more -- 72.32kg is the TWR=1.0 break-even point, effectively the
    # max viable fuel load).
    # vy=-15 solved vertical braking but exhausted the full fuel budget doing
    # it (fuel left = 0.0kg in 8/10 eval episodes), leaving nothing for
    # horizontal correction -- a fuel/TWR resource-contention problem, not a
    # reward-shaping bug. vy=-10 reduces the braking fuel demand; x_range is
    # also tightened 80->50 so less horizontal correction is needed on
    # average, easing demand on both axes simultaneously.
    # That fixed fuel, but vx then converged to a direction-blind lock
    # (constant ~20-28 m/s magnitude regardless of position error, ignoring
    # vx_desired entirely) -- persisted through 4 rounds of reward-shaping and
    # exploration-noise fixes without resolving. Working theory: gimbal is a
    # single actuator (nozzle_angle = angle + gimbal*MAX_GIMBAL_ANGLE) driving
    # BOTH torque (attitude) and the ax/ay thrust split (horizontal/vertical)
    # simultaneously -- at pd_gain=0.3 the built-in PD stabilizer supplies
    # little of the gimbal command, so the agent's own action has to cover
    # attitude stabilization too, competing with horizontal correction on the
    # same scalar output. pd_gain raised 0.3->0.4 here to give the PD term
    # more of that budget back, freeing the agent's action to focus on vx.
    9: dict(pad=20.0, alt=500.0,  vy=-10.0, x_range=50.0,  vx_range=5.0, angle_range=0.17, pd_gain=0.4, fuel=72.0),
    # Stage 10 (was 9, "new"): same as stage 9 but pd_gain=0.35 -- splits the
    # step back down to the old pd_gain=0.3 into two increments instead of
    # one, since stage 11 (wind) needs pd_gain=0.3 and dropping straight from
    # 0.4 would reintroduce the original attitude-authority cliff this fix is
    # meant to avoid.
    10: dict(pad=20.0, alt=500.0,  vy=-10.0, x_range=50.0,  vx_range=5.0, angle_range=0.17, pd_gain=0.35, fuel=72.0),
    # Stages 11-15: introduce horizontal wind gusts (wind_mag = peak acceleration m/s²).
    # Gusts are random bursts: 0.5-1 s on, 2-3 s off.  wind_x is added to obs so the
    # agent can react; the renderer draws a live arrow at the top of the screen.
    #
    # Stage 11 (new): wind-isolation pretraining, same idea as stage 7's
    # vx-isolation -- old stage 11 (wind_mag=5.0 at the full x_range=80/
    # vy=-15/fuel=72 combined difficulty) stalled at 0% success for ~19.4M
    # steps with no code bug found: diagnostics stayed normal throughout,
    # rewards were properly negative (no capped-penalty-style exploit like
    # vx_overspeed), and a verbose per-episode check showed a coherent,
    # repeatable policy (vy perfectly controlled, vx consistently 29-49
    # magnitude in BOTH directions -- not a lock, just insufficient
    # correction authority against gusts that can outrun vx_desired's
    # ~4 m/s max guidance speed at this x_range). Isolates wind reaction
    # from the wide-spawn correction problem by narrowing x_range to 20
    # (comparable to pad width, so spawn error is small) while introducing
    # wind_mag=5.0 at full strength -- vy/fuel/pd_gain unchanged since
    # those are already mastered. If this passes cleanly, wind-compensation
    # is learnable and the combined difficulty (old stage 11, now 12) just
    # needed it decomposed first; if it also stalls, the gust magnitude/
    # duration itself may need revisiting, independent of spawn range.
    11: dict(pad=20.0, alt=500.0,  vy=-15.0, x_range=20.0,  vx_range=5.0, angle_range=0.17, pd_gain=0.3, fuel=72.0,  wind_mag=5.0),
    # Stage 12 (was 11): the full combined difficulty -- wide x_range=80
    # spawn AND wind_mag=5.0 together, now approached via stage 11's
    # isolated wind skill instead of cold.
    # Stage 12 keeps vy=-15 (matching stages 9/10) -- wind is the new challenge here,
    # not descent speed; vy=-20 resumes from stage 13 onward.
    # Extra fuel (stages 13-15) compensates for the lateral corrections wind forces
    # require -- note this trades away hover margin (TWR<1 at 80kg+), which is a
    # known, deliberate tension for those later/harder stages, not an oversight.
    12: dict(pad=20.0, alt=500.0,  vy=-15.0, x_range=80.0,  vx_range=5.0, angle_range=0.17, pd_gain=0.3, fuel=72.0,  wind_mag=5.0),
    13: dict(pad=10.0, alt=500.0,  vy=-20.0, x_range=80.0,  vx_range=5.0, angle_range=0.17, pd_gain=0.3, fuel=80.0,  wind_mag=10.0),
    14: dict(pad=10.0, alt=750.0,  vy=-20.0, x_range=80.0,  vx_range=5.0, angle_range=0.17, pd_gain=0.3, fuel=90.0,  wind_mag=15.0),
    15: dict(pad=5.0,  alt=1000.0, vy=-20.0, x_range=100.0, vx_range=5.0, angle_range=0.17, pd_gain=0.3, fuel=110.0, wind_mag=20.0),
}
MAX_STAGE = max(STAGES.keys())


class RocketLandingEnv(gym.Env):
    """
    Observation (9 values):
        0  x         horizontal position   (m)
        1  y         altitude              (m)
        2  vx        horizontal velocity   (m/s)
        3  vy        vertical velocity     (m/s)
        4  angle     body tilt             (rad)
        5  ang_vel   angular velocity      (rad/s)
        6  fuel      remaining fuel        (kg)
        7  throttle  current throttle      (0-1)
        8  wind_x    wind acceleration     (m/s²)  -- 0 for stages 1-7

    Action (2 values, all stages):
        0  throttle   [0, 1]
        1  gimbal_cmd [-1, 1]  -- agent's steering correction; combined with PD
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 20}

    def __init__(self, stage: int = 1, render_mode=None):
        super().__init__()
        self.stage       = stage
        self.render_mode = render_mode
        self._state      = None
        self._steps      = 0
        self._phi_prev   = 0.0

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        obs_low  = np.array([-500,    0, -50, -200, -np.pi, -5, 0,        0, -MAX_WIND], dtype=np.float32)
        obs_high = np.array([ 500, 1200,  50,   50,  np.pi,  5, MAX_FUEL, 1,  MAX_WIND], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def set_stage(self, stage: int):
        self.stage = stage  # action space unchanged -- always 2D

    def _cfg(self):
        return STAGES[self.stage]

    def _get_obs(self):
        s = self._state
        return np.array(
            [s["x"], s["y"], s["vx"], s["vy"],
             s["angle"], s["ang_vel"], s["fuel"], s["throttle"], s["wind_force"]],
            dtype=np.float32,
        )

    def _total_mass(self):
        return ROCKET_MASS + self._state["fuel"]

    def _phi(self):
        """
        Potential function for PBRS.
        Encodes: weighted distance to pad-centre, speed-profile error, tilt.
        Lower = worse; improving state -> positive shaping reward.
        """
        s   = self._state
        cfg = self._cfg()

        # Separate x and y terms so horizontal guidance has constant strength
        # regardless of altitude. The old sqrt(x²+(0.5y)²) was dominated by y
        # at high altitude, giving ~+380 shaping just for descending and making
        # the x gradient invisible — agent learned vy control but ignored vx/x.
        x_term = abs(s["x"])
        y_term = s["y"]

        # coefficient=1.0 gives v_target=-3.16 m/s at y=10m (in -3 to -5 range),
        # and caps at -5 m/s above y=25m.  Old coefficient=0.3 capped only at
        # y=278m and gave -0.95 m/s at y=10m — unreachably strict for a 500m run.
        v_target  = -min(5.0, 1.0 * np.sqrt(max(s["y"], 1.0)))
        vx_desired = float(np.clip(-0.05 * s["x"], -15.0, 15.0))
        v_err = abs(s["vy"] - v_target) + abs(s["vx"] - vx_desired)

        # Ramp referenced to 150m so factor>=3x throughout the sub-100m zone.
        # Old reference of 100m only reached 3x at y=50m; between y=50-100m the
        # weight was sub-3x, leaving a weak braking signal at 100-50m.
        altitude_factor = 1.0 + 8.0 * max(0.0, 1.0 - s["y"] / 150.0)
        v_err_weight    = 4.0 * altitude_factor

        # No per-step fuel conservation term: the terminal fuel_bonus (+0.5*fuel at
        # landing) is sufficient incentive for efficiency.  The old +0.5*fuel/step
        # penalised every kg burned and suppressed throttling in stage 7 where the
        # agent must burn ~22 kg before gaining deceleration authority.
        # x_term weight raised 4.5->6.0->8.0 to strengthen horizontal position
        # signal relative to vy's three dense per-step bonus channels (see
        # step()). 6.0 wasn't enough gradient to escape the x=-180m attractor
        # the policy converged to.
        return -8.0 * x_term - 0.5 * y_term - v_err_weight * v_err - 5.0 * abs(s["angle"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self._cfg()
        rng = self.np_random
        self._state = {
            "x":           float(rng.uniform(-cfg["x_range"], cfg["x_range"])),
            "y":           cfg["alt"],
            "vx":          float(rng.uniform(-cfg["vx_range"], cfg["vx_range"])),
            "vy":          cfg["vy"],
            "angle":       float(rng.uniform(-cfg["angle_range"], cfg["angle_range"])),
            "ang_vel":     0.0,
            "fuel":        cfg.get("fuel", FUEL_CAPACITY),
            "throttle":    0.0,
            "gimbal":      0.0,
            # Wind state (zero for stages without wind_mag key)
            "wind_force":  0.0,   # current horizontal wind acceleration (m/s²)
            "wind_active": False, # True while a gust is in progress
            "wind_timer":  int(rng.uniform(40, 61)),  # steps until first gust (2-3 s)
        }
        self._steps    = 0
        self._phi_prev = self._phi()   # Phi(s0)
        return self._get_obs(), {}

    def step(self, action):
        cfg = self._cfg()

        throttle     = float(np.clip(action[0], 0.0, 1.0))
        agent_gimbal = float(np.clip(action[1], -1.0, 1.0))

        if self._state["fuel"] <= 0.0:
            throttle = 0.0

        # Residual gimbal: PD stabilises attitude, agent adds steering correction.
        pd_val = float(np.clip(
            self._state["angle"] * 3.0 + self._state["ang_vel"] * 1.0,
            -1.0, 1.0,
        ))
        desired_gimbal = float(np.clip(
            pd_val * cfg["pd_gain"] + agent_gimbal,
            -1.0, 1.0,
        ))

        # Slew-rate limit: prevents bang-bang gimbal chattering at 20 Hz.
        prev_gimbal = self._state["gimbal"]
        gimbal = float(np.clip(
            desired_gimbal,
            prev_gimbal - MAX_GIMBAL_SLEW,
            prev_gimbal + MAX_GIMBAL_SLEW,
        ))

        self._state["throttle"] = throttle
        self._state["gimbal"]   = gimbal
        s = self._state

        prev_vy = s["vy"]   # captured before physics for vy-improvement bonus
        prev_vx = s["vx"]   # captured before physics for vx-improvement bonus

        # ---- Wind gust update ------------------------------------------------
        wind_mag = cfg.get("wind_mag", 0.0)
        if wind_mag > 0.0:
            s["wind_timer"] -= 1
            if s["wind_timer"] <= 0:
                if s["wind_active"]:
                    # Gust ends; wait 2-3 s (40-60 steps) before the next one
                    s["wind_force"]  = 0.0
                    s["wind_active"] = False
                    s["wind_timer"]  = int(self.np_random.uniform(40, 61))
                else:
                    # New gust: random direction and magnitude, lasts 0.5-1 s
                    s["wind_force"]  = float(self.np_random.uniform(-wind_mag, wind_mag))
                    s["wind_active"] = True
                    s["wind_timer"]  = int(self.np_random.uniform(10, 21))

        # ---- Physics ---------------------------------------------------------
        mass         = self._total_mass()
        nozzle_angle = s["angle"] + gimbal * MAX_GIMBAL_ANGLE
        thrust_force = ENGINE_THRUST * throttle

        ax = -thrust_force * np.sin(nozzle_angle) / mass - 0.02 * s["vx"] + s["wind_force"]
        ay =  thrust_force * np.cos(nozzle_angle) / mass - GRAVITY

        gimbal_defl = gimbal * MAX_GIMBAL_ANGLE
        torque      = -thrust_force * np.sin(gimbal_defl) * GIMBAL_ARM
        ang_accel   = torque / MOMENT_OF_INERTIA

        s["ang_vel"] = s["ang_vel"] * ANGULAR_DAMPING + ang_accel * DT
        s["angle"]  += s["ang_vel"] * DT
        s["vx"]     += ax * DT
        s["vy"]     += ay * DT
        s["x"]      += s["vx"] * DT
        s["y"]      += s["vy"] * DT
        s["fuel"]    = max(0.0, s["fuel"] - FUEL_BURN_RATE * throttle * DT)
        self._steps += 1

        # ---- Termination -----------------------------------------------------
        terminated = False
        truncated  = False
        reward     = 0.0

        if s["y"] > cfg["alt"] * 1.5:
            terminated = True
            reward     = -200.0

        elif abs(s["angle"]) > MAX_TILT:
            terminated = True
            reward     = -200.0

        elif s["y"] <= 0.0:
            s["y"]     = 0.0
            terminated = True
            speed_ok   = abs(s["vy"]) <= MAX_LANDING_VY and abs(s["vx"]) <= MAX_LANDING_VX
            upright    = abs(s["angle"]) <= MAX_TILT
            on_pad     = abs(s["x"]) <= cfg["pad"]

            if speed_ok and upright and on_pad:
                # Graded landing bonus: encourages centring, soft touchdown, fuel efficiency.
                centring_bonus = 20.0 * max(0.0, 1.0 - abs(s["x"]) / cfg["pad"])
                speed_bonus    = 20.0 * max(0.0, 1.0 - abs(s["vy"]) / MAX_LANDING_VY)
                fuel_bonus     = 0.5  * s["fuel"]
                reward         = 200.0 + centring_bonus + speed_bonus + fuel_bonus
            elif speed_ok and upright:
                reward = 0.0
            else:
                # Extra penalty scaled to how far vy exceeds the soft-landing threshold.
                # vy=-10 -> -225, vy=-25 -> -300. Creates a gradient: crashing slowly
                # is always better than crashing fast, even when landing is not achievable.
                vy_excess = max(0.0, abs(s["vy"]) - MAX_LANDING_VY)
                reward    = -200.0 - 5.0 * vy_excess

        elif s["fuel"] <= 0.0:
            terminated = True
            reward     = -150.0

        elif self._steps >= MAX_STEPS:
            truncated = True
            reward    = -50.0

        else:
            # Potential-based shaping: r = Phi(s') - Phi(s)
            # Using gamma=1 here so the sum telescopes exactly to Phi(s_T) - Phi(s_0),
            # which is bounded by |Phi_max - Phi_min| regardless of episode length.
            # The gamma<1 formula (gamma*Phi(s') - Phi(s)) pays (gamma-1)*Phi per step
            # as a bonus, which compounds over long episodes and can dominate terminals.
            phi_next       = self._phi()
            reward         = phi_next - self._phi_prev
            self._phi_prev = phi_next

            # Dense braking bonus: explicit reward for each step vy moves toward v_target.
            # PBRS already encodes this via the v_err term in Phi, but the signal is mixed
            # with x and angle terms.  This additive term isolates the vertical braking
            # gradient so it can't be masked by other Phi components.
            v_target_step  = -min(5.0, 1.0 * np.sqrt(max(s["y"], 1.0)))
            vy_improvement = abs(prev_vy - v_target_step) - abs(s["vy"] - v_target_step)
            if vy_improvement > 0.0:
                reward += 3.0 * vy_improvement

            # Direct vy speed reward: penalty proportional to how much vy exceeds
            # v_target (too fast), 0 when on-target or slower.  This is a per-state
            # penalty, not a delta, giving a constant gradient toward the speed profile.
            # Note: formula uses abs(v_target_step) as denominator; user's original
            # divides by v_target (negative), which inverts the sign and penalises
            # slow descent instead of fast descent.
            vy_ratio = (s["vy"] - v_target_step) / abs(v_target_step)
            reward  += 2.0 * float(np.clip(vy_ratio, -1.0, 0.0))

            # On-speed bonus: positive reward when vy is within 20% of v_target.
            # The penalty above pushes away from overspeeding; this pulls toward the
            # target speed profile with a positive signal so the agent actively seeks
            # the zone rather than just avoiding the penalty zone boundary.
            if abs(s["vy"] - v_target_step) <= 0.2 * abs(v_target_step):
                reward += 2.0

            # Dense horizontal-speed shaping, alongside the vy bonuses above.
            # Without this, vy had three dense per-step channels (improvement bonus,
            # overspeed penalty, on-speed bonus) while vx only appeared inside the
            # shared v_err term in Phi -- the agent learned to nail vy and ignored vx.
            #
            # vx_correction is BIDIRECTIONAL (unlike vy_improvement's one-sided
            # max(0,...)): positive when |vx-vx_desired| shrinks, negative when it
            # grows. A one-sided version let the agent drift the wrong way for free
            # (observed: x=-180m, vx_desired=+9, actual vx=-44 -- moving further
            # from the target cost nothing beyond the shared Phi term, which wasn't
            # enough gradient to escape it). This actively punishes wrong-direction
            # drift instead of merely failing to reward it.
            vx_desired_step = float(np.clip(-0.05 * s["x"], -15.0, 15.0))
            vx_correction   = abs(prev_vx - vx_desired_step) - abs(s["vx"] - vx_desired_step)
            reward += 3.0 * vx_correction

            # Magnitude-based overspeed penalty: fires whenever |vx| exceeds
            # |vx_desired|, regardless of vx's sign. Unlike vy (always negative,
            # so a signed ratio works), vx_desired flips sign depending on which
            # side of the pad the rocket is on, so a magnitude comparison is the
            # correct adaptation rather than a literal sign-based mirror of the
            # vy overspeed penalty.
            #
            # Uncapped (was clip(...,0,1), i.e. bottomed out at -2.0/step
            # regardless of overshoot size): a vx of -42 against a desired +7
            # paid the exact same per-step penalty as being 2x over target,
            # so a smooth, well-controlled vy descent (near-constant +2/step
            # on-speed bonus over an ~800-900 step episode, +1400-1800 total)
            # could outweigh an enormous, wrong-direction vx overshoot and
            # still net large positive reward without ever landing -- a real
            # local optimum, confirmed via a locked stage-7 checkpoint scoring
            # +2071 avg reward at 0/50 success. vy's analogous crash-case
            # penalty (vy_excess below) was already uncapped; this matches it.
            vx_overspeed = max(0.0, (abs(s["vx"]) - abs(vx_desired_step)) / max(abs(vx_desired_step), 1.0))
            reward -= 2.0 * float(vx_overspeed)

            # On-speed bonus for vx matching vx_desired. Uses max(|vx_desired|, 1.0)
            # as the tolerance base (unlike the vy version's bare abs(v_target_step))
            # because vx_desired can legitimately be 0 (at x=0, pad centre) -- a bare
            # 20%-of-target tolerance would shrink to a zero-width window exactly
            # where it matters most.
            if abs(s["vx"] - vx_desired_step) <= 0.2 * max(abs(vx_desired_step), 1.0):
                reward += 2.0

        info = {
            "altitude":   s["y"],
            "vy":         s["vy"],
            "vx":         s["vx"],
            "angle_deg":  np.degrees(s["angle"]),
            "fuel_left":  s["fuel"],
            "on_pad":     abs(s["x"]) <= cfg["pad"],
            "stage":      self.stage,
            "wind_force": s["wind_force"],
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        s   = self._state
        bar = 15
        thr = "#" * int(s["throttle"] * bar) + "." * (bar - int(s["throttle"] * bar))
        fp  = s["fuel"] / FUEL_CAPACITY
        fu  = "#" * int(fp * bar) + "." * (bar - int(fp * bar))
        wind_str = f"  wind={s['wind_force']:+5.1f}" if s.get("wind_force", 0.0) != 0.0 else ""
        print(
            f"\r  [stage {self.stage}]  alt={s['y']:6.1f}m  "
            f"x={s['x']:+6.1f}m  vy={s['vy']:+6.1f}  "
            f"angle={np.degrees(s['angle']):+5.1f}deg  "
            f"thr=[{thr}]  fuel=[{fu}] {s['fuel']:4.1f}kg{wind_str}",
            end="", flush=True,
        )
