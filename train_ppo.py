"""
PPO training script with automatic curriculum.

The CurriculumCallback monitors success rate over a rolling window of eval
episodes. When it hits ADVANCE_THRESHOLD it advances all envs to the next
stage. Stage 6 adds a second action (gimbal), which requires rebuilding the
PPO model — the callback handles this transparently.

Usage
-----
    python rocket/train_ppo.py                   # train from scratch (starts at stage 1)
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

# ── Paths ─────────────────────────────────────────────────────────────────────
SAVE_DIR   = "rocket/checkpoints"
LOG_DIR    = "rocket/logs"
BEST_MODEL = "rocket/best_model"
NORM_STATS = "rocket/vec_normalize.pkl"
STAGE_FILE = "rocket/current_stage.txt"   # persists stage across runs

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ── Curriculum settings ───────────────────────────────────────────────────────
ADVANCE_THRESHOLD = 0.80   # success rate needed to move to next stage
EVAL_WINDOW       = 20     # episodes to average over when checking success rate
EVAL_FREQ         = 20_000 # env steps between evaluations
N_ENVS            = 8


# ── Stage helpers ─────────────────────────────────────────────────────────────

def load_stage() -> int:
    if os.path.exists(STAGE_FILE):
        with open(STAGE_FILE) as f:
            return int(f.read().strip())
    return 1

def save_stage(stage: int):
    with open(STAGE_FILE, "w") as f:
        f.write(str(stage))


# ── Env factories ─────────────────────────────────────────────────────────────

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
        clip_reward=10.0,
        gamma=0.99,
        training=training,
    )
    return vec


# ── Curriculum callback ───────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """
    Every EVAL_FREQ steps, runs EVAL_WINDOW deterministic episodes and checks
    success rate. If >= ADVANCE_THRESHOLD, advances to the next stage.

    Stage 6 adds gimbal (2nd action dim). Because PPO's policy network is
    tied to the action space, we rebuild the model at that transition, copying
    over the shared observation encoder weights so we don't lose everything.
    """

    def __init__(self, train_env: VecNormalize, start_stage: int):
        super().__init__()
        self.train_env   = train_env
        self.stage       = start_stage
        self._next_eval  = EVAL_FREQ
        self._ep_results = []   # list of bools: True = landed

    def _on_training_start(self) -> None:
        # When resuming from a checkpoint, num_timesteps already reflects the
        # steps done in prior runs. Anchor the next eval to that, otherwise
        # eval fires on almost every step until _next_eval catches up.
        self._next_eval = self.num_timesteps + EVAL_FREQ

    # ── Evaluation ───────────────────────────────────────────────────────────

    def _run_eval(self) -> float:
        """Run EVAL_WINDOW episodes; return success rate."""
        raw = RocketLandingEnv(stage=self.stage)
        results = []

        for _ in range(EVAL_WINDOW):
            obs, _ = raw.reset()
            done   = False
            while not done:
                obs_n  = self.train_env.normalize_obs(obs.reshape(1, -1))[0]
                action, _ = self.model.predict(obs_n, deterministic=True)
                obs, _, terminated, truncated, info = raw.step(action)
                done = terminated or truncated
            cfg     = STAGES[self.stage]
            speed_ok = abs(info["vy"]) <= 5.0 and abs(info["vx"]) <= 3.0
            on_pad   = abs(raw._state["x"]) <= cfg["pad"]
            results.append(speed_ok and on_pad)

        raw.close()
        return float(np.mean(results))

    # ── Stage advance ─────────────────────────────────────────────────────────

    def _advance_stage(self):
        self.stage += 1
        save_stage(self.stage)
        cfg = STAGES[self.stage]
        print(f"\n  >>> STAGE {self.stage}: pad={cfg['pad']}m  "
              f"alt={cfg['alt']}m  vy={cfg['vy']}  gimbal={cfg['gimbal']}")

        # Rebuild all envs at the new stage
        new_vec = make_vec_env(make_env_fn(self.stage), n_envs=N_ENVS)
        new_vec = VecNormalize(
            new_vec,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
        # Transfer normalisation statistics so we don't reset the obs scale
        new_vec.obs_rms  = self.train_env.obs_rms
        new_vec.ret_rms  = self.train_env.ret_rms
        self.train_env   = new_vec

        if cfg["gimbal"]:
            # Stage 6: action space changes — must rebuild model
            print("  >>> Rebuilding model for 2D action space (gimbal unlocked)")
            new_model = PPO(
                policy="MlpPolicy",
                env=new_vec,
                policy_kwargs=dict(
                    net_arch=[128, 128],
                    activation_fn=__import__("torch").nn.Tanh,
                ),
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log=LOG_DIR,
                verbose=1,
            )
            # Copy shared feature-extractor weights from old model
            try:
                import torch
                old_sd  = self.model.policy.state_dict()
                new_sd  = new_model.policy.state_dict()
                for k in new_sd:
                    if k in old_sd and old_sd[k].shape == new_sd[k].shape:
                        new_sd[k] = old_sd[k]
                new_model.policy.load_state_dict(new_sd)
                print("  >>> Transferred compatible weights from stage-5 model")
            except Exception as e:
                print(f"  >>> Weight transfer skipped: {e}")

            self.model = new_model
        else:
            self.model.set_env(new_vec)
            # No manual buffer reset needed: SB3's collect_rollouts() already
            # calls rollout_buffer.reset() at the start of every rollout. Doing
            # it here mid-rollout desyncs pos/full from the collection loop's
            # own step counter, so train() later asserts on a non-full buffer.

    # ── Callback hook ─────────────────────────────────────────────────────────

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True

        self._next_eval += EVAL_FREQ
        success_rate = self._run_eval()

        print(f"\n  step {self.num_timesteps:>8,}  |  stage {self.stage}  "
              f"|  success rate: {success_rate*100:.0f}%")

        if success_rate >= ADVANCE_THRESHOLD and self.stage < MAX_STAGE:
            self._advance_stage()

        return True


# ── PPO builder ───────────────────────────────────────────────────────────────

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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def find_latest_checkpoint():
    """Return path (without .zip) to the highest-step checkpoint in SAVE_DIR, or None."""
    if not os.path.isdir(SAVE_DIR):
        return None
    best_path, best_steps = None, -1
    for name in os.listdir(SAVE_DIR):
        m = re.match(r"ppo_rocket_(\d+)_steps\.zip$", name)
        if m and int(m.group(1)) > best_steps:
            best_steps = int(m.group(1))
            best_path = os.path.join(SAVE_DIR, name[:-4])
    return best_path


def train(total_steps: int):
    stage     = load_stage()
    train_env = build_vec_env(stage, n_envs=N_ENVS, training=True)

    # Load existing model/norm stats if available
    if os.path.exists(f"{BEST_MODEL}.zip") and os.path.exists(NORM_STATS):
        print(f"Resuming stage {stage} from saved model …")
        model = PPO.load(BEST_MODEL, env=train_env)
        saved_vec = VecNormalize.load(NORM_STATS, make_vec_env(make_env_fn(stage), n_envs=N_ENVS))
        train_env.obs_rms = saved_vec.obs_rms
        train_env.ret_rms = saved_vec.ret_rms
    elif find_latest_checkpoint():
        ckpt = find_latest_checkpoint()
        print(f"No best_model/vec_normalize found. Resuming stage {stage} from "
              f"latest checkpoint {ckpt}.zip (observation normalization stats "
              f"will re-adapt from scratch) …")
        model = PPO.load(ckpt, env=train_env)
    else:
        print(f"Starting fresh from stage {stage} …")
        model = build_model(train_env, stage)

    model.verbose = 1  # PPO.load() restores the saved verbose value; force console output on

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


# ── Evaluation ────────────────────────────────────────────────────────────────

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

    model   = PPO.load(BEST_MODEL)
    cfg     = STAGES[stage]
    landings = 0

    print(f"Evaluating stage {stage}: pad={cfg['pad']}m  alt={cfg['alt']}m  "
          f"gimbal={'on' if cfg['gimbal'] else 'auto'}\n")

    for ep in range(n_episodes):
        obs, _ = raw_env.reset()
        done   = False
        total_r = 0.0
        info   = {}
        while not done:
            obs_n  = vec_env.normalize_obs(obs.reshape(1, -1))[0]
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

        print(f"  Episode {ep+1:2d}: {outcome:>14s}  |  reward {total_r:+8.1f}  "
              f"|  vy={vy:+5.1f}  fuel left={fuel:.1f}kg")

    raw_env.close()
    print(f"\nSuccess rate: {landings}/{n_episodes} ({100*landings/n_episodes:.0f}%)")


# ── Entry point ───────────────────────────────────────────────────────────────

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
