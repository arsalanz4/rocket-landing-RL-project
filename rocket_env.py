"""
2D Rocket Landing Environment with curriculum support.

The env accepts a `stage` parameter (1-6) that controls difficulty.
Stage is set externally by the CurriculumCallback in train_ppo.py.

Stages
------
  1  pad=40m  alt=200m  vy=-10  x=±30m   throttle-only  (current level)
  2  pad=30m  alt=200m  vy=-10  x=±30m   throttle-only
  3  pad=20m  alt=200m  vy=-10  x=±40m   throttle-only
  4  pad=20m  alt=350m  vy=-15  x=±50m   throttle-only
  5  pad=20m  alt=500m  vy=-20  x=±50m   throttle-only
  6  pad=20m  alt=500m  vy=-20  x=±50m   throttle + gimbal (full problem)

Coordinate system
-----------------
  x     : horizontal position (metres), positive = right
  y     : altitude (metres), positive = up
  vx    : horizontal velocity (m/s)
  vy    : vertical velocity (m/s), negative = falling
  angle : rocket body tilt from vertical (radians), positive = tilted right
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Physical constants (fixed across all stages) ──────────────────────────────
GRAVITY           = 9.81
ROCKET_MASS       = 50.0        # kg dry
FUEL_CAPACITY     = 60.0        # kg
ENGINE_THRUST     = 1200.0      # N at full throttle
FUEL_BURN_RATE    = 2.0         # kg/s at full throttle

MAX_GIMBAL_ANGLE  = np.radians(20)
MOMENT_OF_INERTIA = 800.0
GIMBAL_ARM        = 0.5
ANGULAR_DAMPING   = 0.94
MAX_TILT          = np.radians(60)

MAX_LANDING_VY    = 5.0
MAX_LANDING_VX    = 3.0
DT                = 0.05
MAX_STEPS         = 2000

# ── Curriculum stage definitions ──────────────────────────────────────────────
# Each entry: (pad_half, altitude, init_vy, x_range, gimbal_learned)
STAGES = {
    1: dict(pad=40.0, alt=200.0, vy=-10.0, x_range=30.0, gimbal=False),
    2: dict(pad=30.0, alt=200.0, vy=-10.0, x_range=30.0, gimbal=False),
    3: dict(pad=20.0, alt=200.0, vy=-10.0, x_range=40.0, gimbal=False),
    4: dict(pad=20.0, alt=350.0, vy=-15.0, x_range=50.0, gimbal=False),
    5: dict(pad=20.0, alt=500.0, vy=-20.0, x_range=50.0, gimbal=False),
    6: dict(pad=20.0, alt=500.0, vy=-20.0, x_range=50.0, gimbal=True),
}
MAX_STAGE = len(STAGES)


class RocketLandingEnv(gym.Env):
    """
    Observation space (8 values):
        0  x        horizontal position  (m)
        1  y        altitude             (m)
        2  vx       horizontal velocity  (m/s)
        3  vy       vertical velocity    (m/s)
        4  angle    body tilt            (rad)
        5  ang_vel  angular velocity     (rad/s)
        6  fuel     remaining fuel       (kg)
        7  throttle current throttle     (0-1)

    Action space:
        Stages 1-5: [throttle]          1D, gimbal auto-stabilised by PD
        Stage 6:    [throttle, gimbal]  2D, agent controls both
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 20}

    def __init__(self, stage: int = 1, render_mode=None):
        super().__init__()
        self.stage       = stage
        self.render_mode = render_mode
        self._state      = None
        self._steps      = 0
        self._build_spaces()

    def _build_spaces(self):
        cfg = STAGES[self.stage]

        if cfg["gimbal"]:
            self.action_space = spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([1.0,  1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.float32(0.0),
                high=np.float32(1.0),
                shape=(1,),
                dtype=np.float32,
            )

        obs_low  = np.array([-500, 0,   -50, -200, -np.pi, -5, 0,             0], dtype=np.float32)
        obs_high = np.array([ 500, 1000, 50,   50,  np.pi,  5, FUEL_CAPACITY, 1], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

    def set_stage(self, stage: int):
        """Called by CurriculumCallback to advance difficulty."""
        self.stage = stage
        self._build_spaces()

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self._cfg()
        rng = self.np_random
        self._state = {
            "x":        float(rng.uniform(-cfg["x_range"], cfg["x_range"])),
            "y":        cfg["alt"],
            "vx":       float(rng.uniform(-3.0, 3.0)),
            "vy":       cfg["vy"],
            "angle":    float(rng.uniform(-0.1, 0.1)),
            "ang_vel":  0.0,
            "fuel":     FUEL_CAPACITY,
            "throttle": 0.0,
            "gimbal":   0.0,
        }
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        cfg = self._cfg()

        throttle = float(np.clip(action[0], 0.0, 1.0))

        if cfg["gimbal"]:
            # Agent controls gimbal directly
            gimbal = float(np.clip(action[1], -1.0, 1.0))
        else:
            # PD auto-stabiliser keeps the rocket upright
            gimbal = float(np.clip(
                self._state["angle"] * 3.0 + self._state["ang_vel"] * 1.0,
                -1.0, 1.0,
            ))

        if self._state["fuel"] <= 0.0:
            throttle = 0.0

        self._state["throttle"] = throttle
        self._state["gimbal"]   = gimbal
        s = self._state

        # ── Physics ───────────────────────────────────────────────────────────
        mass          = self._total_mass()
        nozzle_angle  = s["angle"] + gimbal * MAX_GIMBAL_ANGLE
        thrust_force  = ENGINE_THRUST * throttle

        ax = -thrust_force * np.sin(nozzle_angle) / mass - 0.02 * s["vx"]
        ay =  thrust_force * np.cos(nozzle_angle) / mass - GRAVITY

        gimbal_defl  = gimbal * MAX_GIMBAL_ANGLE
        torque       = -thrust_force * np.sin(gimbal_defl) * GIMBAL_ARM
        ang_accel    = torque / MOMENT_OF_INERTIA

        s["ang_vel"] = s["ang_vel"] * ANGULAR_DAMPING + ang_accel * DT
        s["angle"]  += s["ang_vel"] * DT
        s["vx"]     += ax * DT
        s["vy"]     += ay * DT
        s["x"]      += s["vx"] * DT
        s["y"]      += s["vy"] * DT
        s["fuel"]    = max(0.0, s["fuel"] - FUEL_BURN_RATE * throttle * DT)
        self._steps += 1

        # ── Termination ───────────────────────────────────────────────────────
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
                reward = 200.0
            elif speed_ok and upright:
                reward = 80.0
            else:
                reward = -200.0

        elif s["fuel"] <= 0.0:
            terminated = True
            reward     = -150.0

        elif self._steps >= MAX_STEPS:
            truncated = True
            reward    = -50.0

        else:
            # ── Shaping reward ────────────────────────────────────────────────
            speed   = np.sqrt(s["vx"]**2 + s["vy"]**2)
            reward -= 0.02 * speed
            reward -= 0.01 * s["y"]
            reward -= 0.05 * abs(s["x"])
            reward -= 0.05 * abs(s["angle"])
            reward -= 0.05 * throttle

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
