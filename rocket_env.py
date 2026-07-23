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
MAX_LANDING_VX    = 3.0
DT                = 0.05
MAX_STEPS         = 2000

# ---- Curriculum stages -------------------------------------------------------
# pd_gain: weight of the PD stabiliser in the residual gimbal formula.
# vx_range / angle_range: random initial horizontal velocity and tilt (stages 4+).
STAGES = {
    1: dict(pad=40.0, alt=150.0, vy=-5.0,  x_range=10.0, vx_range=0.0, angle_range=0.05, pd_gain=1.0),
    2: dict(pad=30.0, alt=200.0, vy=-10.0, x_range=25.0, vx_range=0.0, angle_range=0.05, pd_gain=1.0),
    3: dict(pad=20.0, alt=250.0, vy=-10.0, x_range=40.0, vx_range=0.0, angle_range=0.05, pd_gain=1.0),
    # Stage 4: increase altitude and speed only — no random vx/angle yet, keep pd_gain high
    4: dict(pad=20.0, alt=350.0, vy=-15.0, x_range=60.0, vx_range=0.0, angle_range=0.05, pd_gain=1.0),
    # Stages 5-7: 500m altitude requires more fuel — 20% extra (72 kg vs 60)
    5: dict(pad=20.0, alt=500.0, vy=-20.0, x_range=80.0, vx_range=5.0, angle_range=0.17, pd_gain=0.7, fuel=72.0),
    6: dict(pad=20.0, alt=500.0, vy=-20.0, x_range=80.0, vx_range=5.0, angle_range=0.17, pd_gain=0.5, fuel=72.0),
    7: dict(pad=20.0, alt=500.0, vy=-20.0, x_range=80.0, vx_range=5.0, angle_range=0.17, pd_gain=0.3, fuel=94.0),
}
MAX_STAGE = len(STAGES)


class RocketLandingEnv(gym.Env):
    """
    Observation (8 values):
        0  x        horizontal position  (m)
        1  y        altitude             (m)
        2  vx       horizontal velocity  (m/s)
        3  vy       vertical velocity    (m/s)
        4  angle    body tilt            (rad)
        5  ang_vel  angular velocity     (rad/s)
        6  fuel     remaining fuel       (kg)
        7  throttle current throttle     (0-1)

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

        obs_low  = np.array([-500,    0, -50, -200, -np.pi, -5, 0,             0], dtype=np.float32)
        obs_high = np.array([ 500, 1000,  50,   50,  np.pi,  5, FUEL_CAPACITY, 1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def set_stage(self, stage: int):
        self.stage = stage  # action space unchanged -- always 2D

    def _cfg(self):
        return STAGES[self.stage]

    def _get_obs(self):
        s = self._state
        return np.array(
            [s["x"], s["y"], s["vx"], s["vy"],
             s["angle"], s["ang_vel"], s["fuel"], s["throttle"]],
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

        # Fuel term: burning 1 kg costs 0.5 shaping per step, encouraging conservation
        # throughout the flight rather than only at landing (fuel_bonus terminal).
        return -4.5 * x_term - 0.5 * y_term - v_err_weight * v_err - 5.0 * abs(s["angle"]) + 0.5 * s["fuel"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self._cfg()
        rng = self.np_random
        self._state = {
            "x":       float(rng.uniform(-cfg["x_range"], cfg["x_range"])),
            "y":       cfg["alt"],
            "vx":      float(rng.uniform(-cfg["vx_range"], cfg["vx_range"])),
            "vy":      cfg["vy"],
            "angle":   float(rng.uniform(-cfg["angle_range"], cfg["angle_range"])),
            "ang_vel": 0.0,
            "fuel":    cfg.get("fuel", FUEL_CAPACITY),
            "throttle": 0.0,
            "gimbal":  0.0,
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

        # ---- Physics ---------------------------------------------------------
        mass         = self._total_mass()
        nozzle_angle = s["angle"] + gimbal * MAX_GIMBAL_ANGLE
        thrust_force = ENGINE_THRUST * throttle

        ax = -thrust_force * np.sin(nozzle_angle) / mass - 0.02 * s["vx"]
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
            # with x, angle, and fuel terms.  This additive term isolates the vertical
            # braking gradient so it can't be masked by other components competing in Phi.
            v_target_step   = -min(5.0, 1.0 * np.sqrt(max(s["y"], 1.0)))
            vy_improvement  = abs(prev_vy - v_target_step) - abs(s["vy"] - v_target_step)
            if vy_improvement > 0.0:
                reward += 3.0 * vy_improvement

        info = {
            "altitude":  s["y"],
            "vy":        s["vy"],
            "vx":        s["vx"],
            "angle_deg": np.degrees(s["angle"]),
            "fuel_left": s["fuel"],
            "on_pad":    abs(s["x"]) <= cfg["pad"],
            "stage":     self.stage,
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
        print(
            f"\r  [stage {self.stage}]  alt={s['y']:6.1f}m  "
            f"x={s['x']:+6.1f}m  vy={s['vy']:+6.1f}  "
            f"angle={np.degrees(s['angle']):+5.1f}deg  "
            f"thr=[{thr}]  fuel=[{fu}] {s['fuel']:4.1f}kg",
            end="", flush=True,
        )
