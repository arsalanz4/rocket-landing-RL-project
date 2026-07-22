"""
PPO training script with automatic curriculum.

The CurriculumCallback monitors success rate over a rolling window of eval
episodes. When it hits ADVANCE_THRESHOLD for REQUIRED_CONSECUTIVE_PASSES
windows in a row, it advances all envs to the next stage.

All stages now share the same 2D action space [throttle, gimbal], so no
model rebuild is ever needed on stage transition.

Usage
-----
    python rocket/train_ppo.py                   # train from scratch (stage 1)
    python rocket/train_ppo.py --steps 5000000   # longer run
    python rocket/train_ppo.py --eval            # evaluate saved model
    python rocket/train_ppo.py --eval --episodes 20
"""

import argparse
import os
import re

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from rocket_env import RocketLandingEnv, STAGES, MAX_STAGE

# ---- Paths -------------------------------------------------------------------
SAVE_DIR   = "rocket/checkpoints"
LOG_DIR    = "rocket/logs"
BEST_MODEL = "rocket/best_model"
NORM_STATS = "rocket/vec_normalize.pkl"
STAGE_FILE = "rocket/current_stage.txt"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ---- Curriculum settings -----------------------------------------------------
ADVANCE_THRESHOLD = {
    1: 0.80,
    2: 0.80,
    3: 0.75,
    4: 0.75,
    5: 0.70,
    6: 0.70,
    7: 0.70,
}
EVAL_WINDOW               = 20
EVAL_FREQ                 = 20_000
N_ENVS                    = 8
REQUIRED_CONSECUTIVE_PASSES = 3


# ---- Stage helpers -----------------------------------------------------------

def load_stage() -> int:
    if os.path.exists(STAGE_FILE):
        with open(STAGE_FILE, encoding="utf-8-sig") as f:  # utf-8-sig strips BOM if present
            content = f.read().strip()
            if content:
                return int(content)
    return 1

def save_stage(stage: int):
    with open(STAGE_FILE, "w", encoding="utf-8") as f:
        f.write(str(stage))


# ---- Env factories -----------------------------------------------------------

def make_env_fn(stage: int):
    def _make():
        return RocketLandingEnv(stage=stage)
    return _make


def build_vec_env(stage: int, n_envs: int = N_ENVS, training: bool = True):
    vec = make_vec_env(make_env_fn(stage), n_envs=n_envs)
    vec = VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=training,
        clip_obs=10.0,
        clip_reward=15.0,   # raised: graded landing bonus can reach ~260
        gamma=0.99,
        training=training,
    )
    return vec


# ---- Curriculum callback -----------------------------------------------------

class CurriculumCallback(BaseCallback):
    """
    Every EVAL_FREQ steps, runs EVAL_WINDOW deterministic episodes and checks
    success rate. Requires REQUIRED_CONSECUTIVE_PASSES windows above threshold
    before advancing -- prevents lucky single-window stage jumps.
    """

    def __init__(self, train_env: VecNormalize, start_stage: int):
        super().__init__()
        self.train_env            = train_env
        self.stage                = start_stage
        self._next_eval           = EVAL_FREQ
        self._consecutive_passes  = 0
        self._pending_advance     = False

    def _on_training_start(self) -> None:
        self._next_eval = self.num_timesteps + EVAL_FREQ

    def _on_rollout_start(self) -> None:
        # Apply any pending stage advance NOW -- buffer just reset by SB3,
        # so there are no stale observations that could cause NaN.
        if self._pending_advance:
            self._pending_advance = False
            self._do_advance_stage()

    def _on_rollout_end(self) -> None:
        # Catch NaN/Inf in the rollout buffer BEFORE train() consumes it.
        buf = self.model.rollout_buffer
        for name in ("observations", "actions", "rewards", "returns", "values", "log_probs", "advantages"):
            arr = getattr(buf, name, None)
            if arr is None:
                continue
            bad = ~np.isfinite(arr)
            if bad.any():
                idx = tuple(np.argwhere(bad)[0])
                raise RuntimeError(
                    f"rollout_buffer.{name} has {bad.sum()}/{arr.size} non-finite values "
                    f"(stage {self.stage}, step {self.num_timesteps}). First bad index "
                    f"{idx} = {arr[idx]!r}. Aborting before train() to pinpoint the source."
                )

    # ---- Evaluation ----------------------------------------------------------

    def _run_eval(self) -> float:
        raw = RocketLandingEnv(stage=self.stage)
        results = []

        for _ in range(EVAL_WINDOW):
            obs, _ = raw.reset()
            done   = False
            while not done:
                obs_n = self.train_env.normalize_obs(obs.reshape(1, -1))[0]
                action, _ = self.model.predict(obs_n, deterministic=True)
                obs, _, terminated, truncated, info = raw.step(action)
                done = terminated or truncated

            cfg      = STAGES[self.stage]
            speed_ok = abs(info["vy"]) <= 5.0 and abs(info["vx"]) <= 3.0
            on_pad   = abs(raw._state["x"]) <= cfg["pad"]
            results.append(speed_ok and on_pad)

        raw.close()
        return float(np.mean(results))

    # ---- Stage advance -------------------------------------------------------

    def _advance_stage(self):
        self._pending_advance = True
        print(f"\n  >>> Stage advance queued (will apply at next rollout start)")

    def _do_advance_stage(self):
        self.stage += 1
        save_stage(self.stage)
        cfg = STAGES[self.stage]
        print(f"\n  >>> STAGE {self.stage}: pad={cfg['pad']}m  "
              f"alt={cfg['alt']}m  vy={cfg['vy']}  pd_gain={cfg['pd_gain']}")

        new_vec = make_vec_env(make_env_fn(self.stage), n_envs=N_ENVS)
        new_vec = VecNormalize(
            new_vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=15.0,
            gamma=0.99,
        )
        # Transfer normalisation statistics so the policy doesn't see a sudden obs scale jump.
        new_vec.obs_rms = self.train_env.obs_rms
        new_vec.ret_rms = self.train_env.ret_rms
        self.train_env  = new_vec
        self.model.set_env(new_vec)
        # set_env() does not reset the env, so _last_obs is None. Reset manually.
        obs = new_vec.reset()
        self.model._last_obs = obs
        self.model._last_episode_starts = np.ones((N_ENVS,), dtype=bool)

    # ---- Main hook -----------------------------------------------------------

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True

        self._next_eval += EVAL_FREQ
        success_rate = self._run_eval()

        threshold = ADVANCE_THRESHOLD[self.stage]
        if success_rate >= threshold:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        print(f"\n  step {self.num_timesteps:>8,}  |  stage {self.stage}  "
              f"|  success {success_rate*100:.0f}%  "
              f"(need {threshold*100:.0f}%, "
              f"{self._consecutive_passes}/{REQUIRED_CONSECUTIVE_PASSES} consecutive)")

        if self._consecutive_passes >= REQUIRED_CONSECUTIVE_PASSES and self.stage < MAX_STAGE:
            self._consecutive_passes = 0
            self._advance_stage()

        return True


# ---- PPO builder -------------------------------------------------------------

def build_model(train_env: VecNormalize, stage: int) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(net_arch=[128, 128], activation_fn=__import__("torch").nn.Tanh),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,    # slightly higher: 2D action space needs more initial exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )


# ---- Training ----------------------------------------------------------------

def find_latest_checkpoint():
    if not os.path.isdir(SAVE_DIR):
        return None
    best_path, best_steps = None, -1
    for name in os.listdir(SAVE_DIR):
        m = re.match(r"ppo_rocket_(\d+)_steps\.zip$", name)
        if m and int(m.group(1)) > best_steps:
            best_steps = int(m.group(1))
            best_path  = os.path.join(SAVE_DIR, name[:-4])
    return best_path


def train(total_steps: int):
    stage     = load_stage()
    train_env = build_vec_env(stage, n_envs=N_ENVS, training=True)

    if os.path.exists(f"{BEST_MODEL}.zip") and os.path.exists(NORM_STATS):
        print(f"Resuming stage {stage} from saved model ...")
        model = PPO.load(BEST_MODEL, env=train_env)
        saved_vec = VecNormalize.load(NORM_STATS, make_vec_env(make_env_fn(stage), n_envs=N_ENVS))
        train_env.obs_rms = saved_vec.obs_rms
        train_env.ret_rms = saved_vec.ret_rms
    elif find_latest_checkpoint():
        ckpt = find_latest_checkpoint()
        print(f"Resuming stage {stage} from checkpoint {ckpt}.zip ...")
        model = PPO.load(ckpt, env=train_env)
    else:
        print(f"Starting fresh from stage {stage} ...")
        model = build_model(train_env, stage)

    model.verbose   = 1
    model.target_kl = 0.03

    curriculum_cb = CurriculumCallback(train_env, stage)

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // N_ENVS,
        save_path=SAVE_DIR,
        name_prefix="ppo_rocket",
    )

    print(f"Training for {total_steps:,} steps  |  current stage: {stage}")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps=total_steps,
        callback=[curriculum_cb, checkpoint_cb],
        reset_num_timesteps=False,
        tb_log_name="PPO_curriculum",
    )

    model.save(BEST_MODEL)
    curriculum_cb.train_env.save(NORM_STATS)
    save_stage(curriculum_cb.stage)
    print(f"\nSaved model -> {BEST_MODEL}.zip  (stage {curriculum_cb.stage})")


# ---- Evaluation --------------------------------------------------------------

def evaluate(n_episodes: int = 10):
    if not os.path.exists(f"{BEST_MODEL}.zip"):
        print("No model found. Run training first.")
        return

    stage   = load_stage()
    raw_env = RocketLandingEnv(stage=stage)

    vec_env = make_vec_env(make_env_fn(stage), n_envs=1)
    vec_env = VecNormalize.load(NORM_STATS, vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = PPO.load(BEST_MODEL)
    cfg   = STAGES[stage]
    landings = 0

    print(f"Evaluating stage {stage}: pad={cfg['pad']}m  alt={cfg['alt']}m  "
          f"pd_gain={cfg['pd_gain']}\n")

    for ep in range(n_episodes):
        obs, _ = raw_env.reset()
        done   = False
        total_r = 0.0
        info   = {}
        while not done:
            obs_n = vec_env.normalize_obs(obs.reshape(1, -1))[0]
            action, _ = model.predict(obs_n, deterministic=True)
            obs, reward, terminated, truncated, info = raw_env.step(action)
            total_r += reward
            done = terminated or truncated

        vy     = info.get("vy", -999)
        fuel   = info.get("fuel_left", 0)
        on_pad = abs(raw_env._state["x"]) <= cfg["pad"]
        speed_ok = abs(vy) <= 5.0 and abs(info.get("vx", 99)) <= 3.0

        if speed_ok and on_pad:
            outcome = "LANDED"
            landings += 1
        elif speed_ok:
            outcome = "soft (off-pad)"
        else:
            outcome = "CRASH"

        vx = info.get("vx", 0.0)
        x_final = raw_env._state["x"]
        print(f"  Episode {ep+1:2d}: {outcome:>14s}  |  reward {total_r:+8.1f}  "
              f"|  vy={vy:+5.1f}  vx={vx:+5.1f}  x={x_final:+6.1f}m  fuel left={fuel:.1f}kg")

    raw_env.close()
    print(f"\nSuccess rate: {landings}/{n_episodes} ({100*landings/n_episodes:.0f}%)")


# ---- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",    type=int, default=3_000_000)
    parser.add_argument("--eval",     action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    if args.eval:
        evaluate(n_episodes=args.episodes)
    else:
        train(total_steps=args.steps)
        print("\n--- Final evaluation ---")
        evaluate(n_episodes=10)
