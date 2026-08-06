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
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from rocket_env import RocketLandingEnv, STAGES, MAX_STAGE, MAX_LANDING_VY, MAX_LANDING_VX

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
    # Stage 0: diagnostic vx-isolation pretraining (see STAGES comment).
    # Stages 1-6: lowered to sprint through known-solvable curriculum after retrain.
    # Stage 7+ restored to original thresholds — these are where real learning happens.
    # Stage 8: pd_gain raised 0.3->0.4 (more built-in attitude authority, see
    # STAGES comment). Stage 9 (new): intermediate pd_gain=0.35 step down to
    # the old 0.3 before wind is introduced. Stages 10-13 are the old 9-12
    # renumbered (wind stages).
    0: 0.80,
    1: 0.65, 2: 0.65, 3: 0.60, 4: 0.60,
    5: 0.60, 6: 0.60, 7: 0.70,
    8: 0.80, 9: 0.80, 10: 0.80, 11: 0.75, 12: 0.70, 13: 0.65,
}
EVAL_WINDOW               = 20
EVAL_FREQ                 = 20_000
N_ENVS                    = 4    # CPU: fewer envs → smaller batches fit in cache better
# Stages 1-6: only 1 consecutive pass needed to advance; stage 7+ keep 3.
# Override in _on_step via SPRINT_STAGES constant below.
REQUIRED_CONSECUTIVE_PASSES = 3
SPRINT_STAGES = {1, 2, 3, 4, 5, 6}  # advance after 1 pass, not 3

# How often to overwrite best_model.zip/vec_normalize.pkl during training (not
# just at the end of a full --steps run). These two files are the only ones
# meant to be pushed to git for cross-machine handoff — keeping them current
# means you can stop mid-run, push, and resume from the same point elsewhere.
SYNC_SAVE_FREQ = 50_000

# Hard bounds on the Gaussian policy's log_std parameter. PPO's clip_range
# bounds how much the policy RATIO can change per update, but nothing bounds
# the absolute value of log_std itself -- it can drift unboundedly over many
# small updates without ever tripping that clip. ent_coef alone isn't a
# reliable guard (a run under ent_coef=0.02 still grew action_std from ~1.0
# at step 50k to ~2.3e13 by step 10.4M, fully collapsing training long before
# the numbers got that extreme). Clamping directly is the standard fix.
# Range [-2.0, 1.0] -> action_std in [~0.135, ~2.72], ample for a [0,1]/[-1,1]
# bounded action space without ever reaching a damaging magnitude.
#
# Ceiling was temporarily raised to 1.5 (std ~4.48) for ~12M steps to try to
# break stage 8's direction-blind vx lock via more exploration noise. Result:
# partial and non-durable -- vx went from 100% one sign to a genuine 50/50
# split, but magnitude stayed ~21-28 regardless of direction and never
# started tracking vx_desired. Reverted back to 1.0; the working theory has
# shifted from "needs more exploration" to "pd_gain=0.3 structurally forces
# the single gimbal actuator to fight between attitude stabilization and
# horizontal correction" -- see STAGES stage 8/9 comments.
LOG_STD_MIN = -2.0
LOG_STD_MAX = 1.0

# Learning rate applied on RESUME only (build_model's 3e-4 still applies to a
# fresh model). Was cut to 1e-4 alongside the x_range 80->60 reduction as a
# low-risk pass at stage 7's ~18.8M-step plateau, but the same tight
# one-directional vx lock that hit the earlier MAX_LANDING_VX widen
# experiment reappeared a third time under this reward-untouched pair of
# changes -- pointing at the step-135M checkpoint region itself being fragile
# rather than any specific intervention. Reverted to 3e-4 (a no-op override,
# matching the original constant rate) to run a clean control test: does this
# checkpoint lock even with ZERO changes, given enough steps? See the paired
# x_range revert in rocket_env.py's stage 7 comment.
RESUME_LEARNING_RATE = 3e-4


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
            speed_ok = abs(info["vy"]) <= MAX_LANDING_VY and abs(info["vx"]) <= MAX_LANDING_VX
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
        required = 1 if self.stage in SPRINT_STAGES else REQUIRED_CONSECUTIVE_PASSES

        if success_rate >= threshold:
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        sprint_tag = " [SPRINT]" if self.stage in SPRINT_STAGES else ""
        print(f"\n  step {self.num_timesteps:>8,}  |  stage {self.stage}{sprint_tag}  "
              f"|  success {success_rate*100:.0f}%  "
              f"(need {threshold*100:.0f}%, "
              f"{self._consecutive_passes}/{required} consecutive)")

        if self._consecutive_passes >= required and self.stage < MAX_STAGE:
            self._consecutive_passes = 0
            self._advance_stage()

        return True


class SyncSaveCallback(BaseCallback):
    """Periodically overwrites best_model.zip/vec_normalize.pkl mid-run, so
    the current state is always ready to push to git for cross-machine
    handoff -- not just when a full --steps run finishes uninterrupted.

    Reads curriculum_cb.train_env dynamically on every save (rather than
    capturing it once at construction) because CurriculumCallback replaces
    train_env with a new object on every stage advance.
    """

    def __init__(self, curriculum_cb: "CurriculumCallback", save_freq: int):
        super().__init__()
        self.curriculum_cb = curriculum_cb
        self.save_freq      = save_freq
        self._next_save      = save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_save:
            self._next_save += self.save_freq
            self.model.save(BEST_MODEL)
            self.curriculum_cb.train_env.save(NORM_STATS)
        return True


class LogStdClampCallback(BaseCallback):
    """Clamps the policy's log_std parameter to [LOG_STD_MIN, LOG_STD_MAX]
    every step, as a hard guard against unbounded action-variance growth --
    see the comment above those constants for why ent_coef tuning alone
    isn't sufficient."""

    def _on_step(self) -> bool:
        with torch.no_grad():
            self.model.policy.log_std.data.clamp_(LOG_STD_MIN, LOG_STD_MAX)
        return True


# ---- PPO builder -------------------------------------------------------------

def build_model(train_env: VecNormalize, stage: int) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs=dict(net_arch=[128, 128], activation_fn=torch.nn.Tanh),
        learning_rate=3e-4,
        n_steps=512,        # CPU: rollout = N_ENVS * n_steps = 4*512 = 2048 samples
        batch_size=128,     # CPU: fits comfortably in L2/L3 cache
        n_epochs=10,
        device="cpu",
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,    # reverted from 0.05: high entropy destroyed horizontal control
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
        model = PPO.load(BEST_MODEL, env=train_env, custom_objects={"learning_rate": RESUME_LEARNING_RATE})
        saved_vec = VecNormalize.load(NORM_STATS, make_vec_env(make_env_fn(stage), n_envs=N_ENVS))
        train_env.obs_rms = saved_vec.obs_rms
        train_env.ret_rms = saved_vec.ret_rms
    elif find_latest_checkpoint():
        ckpt = find_latest_checkpoint()
        print(f"Resuming stage {stage} from checkpoint {ckpt}.zip ...")
        model = PPO.load(ckpt, env=train_env, custom_objects={"learning_rate": RESUME_LEARNING_RATE})
        ckpt_dir, ckpt_name = os.path.split(ckpt)
        vecnorm_path = os.path.join(ckpt_dir, ckpt_name.replace("ppo_rocket_", "ppo_rocket_vecnormalize_", 1) + ".pkl")
        if os.path.exists(vecnorm_path):
            print(f"  ... with matched VecNormalize stats from {vecnorm_path}")
            saved_vec = VecNormalize.load(vecnorm_path, make_vec_env(make_env_fn(stage), n_envs=N_ENVS))
            train_env.obs_rms = saved_vec.obs_rms
            train_env.ret_rms = saved_vec.ret_rms
        else:
            print("  ... no matched VecNormalize found for this checkpoint, starting fresh stats")
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
        save_vecnormalize=True,
    )

    sync_cb    = SyncSaveCallback(curriculum_cb, save_freq=SYNC_SAVE_FREQ)
    logstd_cb  = LogStdClampCallback()

    # Clamp immediately on resume too, in case the loaded checkpoint's
    # log_std is already out of range (as it will be when recovering from
    # a prior runaway-std run).
    with torch.no_grad():
        model.policy.log_std.data.clamp_(LOG_STD_MIN, LOG_STD_MAX)

    print(f"Training for {total_steps:,} steps  |  current stage: {stage}")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps=total_steps,
        callback=[curriculum_cb, checkpoint_cb, sync_cb, logstd_cb],
        reset_num_timesteps=False,
        tb_log_name="PPO_curriculum",
    )

    model.save(BEST_MODEL)
    curriculum_cb.train_env.save(NORM_STATS)
    save_stage(curriculum_cb.stage)
    print(f"\nSaved model -> {BEST_MODEL}.zip  (stage {curriculum_cb.stage})")


# ---- Evaluation --------------------------------------------------------------

# Default eval size/seed. Raised 10->50 and given a fixed default seed after
# this session's plateau investigation found 10-episode unseeded evals
# unreliable for before/after comparisons -- this policy's outcome
# distribution is wide and bimodal (near-perfect landing OR a large miss),
# so a small unseeded sample can land anywhere in that range purely by
# chance, making two 10-episode evals of the SAME unchanged model look like
# different states when nothing actually changed. 50 episodes with a fixed
# seed gives a stable, reproducible read and makes before/after comparisons
# actually comparable.
DEFAULT_EVAL_EPISODES = 50
DEFAULT_EVAL_SEED     = 42


def evaluate(n_episodes: int = DEFAULT_EVAL_EPISODES, seed: int | None = DEFAULT_EVAL_SEED):
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
          f"pd_gain={cfg['pd_gain']}  seed={seed}\n")

    for ep in range(n_episodes):
        # Seed only the first episode's reset -- Gymnasium's reset(seed=...)
        # reseeds the env's RNG; leaving seed=None on later episodes lets it
        # keep evolving deterministically from that seed, so the FULL
        # sequence of spawn conditions across all n_episodes is fixed and
        # reproducible run-to-run given the same top-level seed.
        obs, _ = raw_env.reset(seed=seed if ep == 0 else None)
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
        speed_ok = abs(vy) <= MAX_LANDING_VY and abs(info.get("vx", 99)) <= MAX_LANDING_VX

        if speed_ok and on_pad:
            outcome = "LANDED"
            landings += 1
        elif speed_ok:
            outcome = "soft (off-pad)"
        else:
            outcome = "CRASH"

        vx = info.get("vx", 0.0)
        x_final  = raw_env._state["x"]
        wind_str = f"  wind={info.get('wind_force', 0.0):+.1f}" if info.get("wind_force", 0.0) != 0.0 else ""
        print(f"  Episode {ep+1:2d}: {outcome:>14s}  |  reward {total_r:+8.1f}  "
              f"|  vy={vy:+5.1f}  vx={vx:+5.1f}  x={x_final:+6.1f}m  fuel left={fuel:.1f}kg{wind_str}")

    raw_env.close()
    print(f"\nSuccess rate: {landings}/{n_episodes} ({100*landings/n_episodes:.0f}%)")


# ---- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",    type=int, default=3_000_000)
    parser.add_argument("--eval",     action="store_true")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--seed",     type=int, default=DEFAULT_EVAL_SEED)
    args = parser.parse_args()

    if args.eval:
        evaluate(n_episodes=args.episodes, seed=args.seed)
    else:
        train(total_steps=args.steps)
        print("\n--- Final evaluation ---")
        evaluate()
