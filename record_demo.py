"""
Record PPO rocket landing demos to MP4 for LinkedIn / sharing.

Usage
-----
  # Video 1: untrained (random) policy — no --model flag
  python rocket/record_demo.py \
      --stage 1 --episodes 10 --output stage1_crashes.mp4 \
      --title "Stage 1 — Untrained Policy"

  # Video 2: trained model — point at best_model.zip once training completes
  python rocket/record_demo.py \
      --stage 7 --episodes 10 --output stage7_landings.mp4 \
      --model rocket/best_model --vec-normalize rocket/vec_normalize.pkl \
      --title "Stage 7 — Trained (90% Success)"

Notes
-----
  - Run from d:\\Documents\\Python\\ (project root), NOT from inside rocket\\
  - Each simulation step (DT=0.05 s) becomes one video frame; video plays at
    SIM_FPS=20 so playback is real-time.
  - Omitting --model uses random actions (untrained behaviour).
  - Omitting --output writes to stage<N>_demo.mp4 in the current directory.
"""

import argparse
import os
import sys

import numpy as np
import pygame
import imageio

sys.path.insert(0, os.path.dirname(__file__))
from rocket_env import (
    RocketLandingEnv, STAGES,
    MAX_LANDING_VY, MAX_LANDING_VX,
    MAX_GIMBAL_ANGLE, MAX_TILT,
)

# ── Display ───────────────────────────────────────────────────────────────────
W, H        = 900, 700
SIM_FPS     = 20        # matches DT=0.05 → real-time playback
WORLD_WIDTH = 300.0     # metres shown horizontally (±150 m)
WORLD_HEIGHT = 200.0    # overridden per stage in record()

# Colours
BG_COL     = (8,   8,   16)
GROUND_COL = (55,  55,  55)
PAD_COL    = (220, 180,  40)
ROCKET_COL = (220, 220, 220)
FLAME_COLS = [(255, 200, 0), (255, 120, 0), (200, 50, 0)]
TRAIL_COL  = (180, 100, 40)
TEXT_COL   = (230, 230, 230)
DIM_COL    = (120, 120, 120)
TITLE_COL  = (190, 200, 255)
CRASH_COL  = (255,  50,  20)
LAND_COL   = (70,  220,  80)
ALT_COL    = (60,   60, 110)
HUD_BG     = (0,    0,   0, 175)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def world_to_screen(x, y):
    px = int(W / 2 + x * (W / WORLD_WIDTH))
    py = int(H   - y * (H / WORLD_HEIGHT))
    return px, py


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_background(surf, pad_half, trail, state):
    surf.fill(BG_COL)

    # Ground slab
    ground_y = world_to_screen(0, 0)[1]
    pygame.draw.rect(surf, GROUND_COL, (0, ground_y, W, H - ground_y))

    # Landing pad + edge markers
    pad_left  = world_to_screen(-pad_half, 0)[0]
    pad_right = world_to_screen( pad_half, 0)[0]
    pygame.draw.rect(surf, PAD_COL, (pad_left, ground_y - 5, pad_right - pad_left, 5))
    for mx in (-pad_half, pad_half):
        sx = world_to_screen(mx, 0)[0]
        pygame.draw.rect(surf, PAD_COL, (sx - 3, ground_y - 14, 6, 14))

    # Exhaust trail (fades by alpha)
    for i, (tx, ty) in enumerate(trail):
        alpha  = int(160 * i / max(len(trail), 1))
        radius = max(1, 3 - (len(trail) - i) // 30)
        ts = pygame.Surface((radius*2+1, radius*2+1), pygame.SRCALPHA)
        pygame.draw.circle(ts, (*TRAIL_COL, alpha), (radius, radius), radius)
        surf.blit(ts, (tx - radius, ty - radius))

    # Faint altitude dashed line
    _, ay = world_to_screen(0, state["y"])
    if 0 < ay < H:
        for x in range(0, W, 20):
            pygame.draw.line(surf, ALT_COL, (x, ay), (x + 10, ay), 1)


def _rotate(pts, cx, cy, a):
    ca, sa = np.cos(a), np.sin(a)
    return [(cx + (px - cx)*ca - (py - cy)*sa,
             cy + (px - cx)*sa + (py - cy)*ca) for px, py in pts]


def draw_rocket(surf, state):
    sx, sy = world_to_screen(state["x"], state["y"])
    angle, gimbal, throttle = state["angle"], state["gimbal"], state["throttle"]
    bw, bh = 14, 36

    body = _rotate([(sx-bw//2, sy), (sx-bw//2, sy-bh),
                    (sx+bw//2, sy-bh), (sx+bw//2, sy)], sx, sy, angle)
    nose = _rotate([(sx, sy-bh-16), (sx-bw//2, sy-bh),
                    (sx+bw//2, sy-bh)],                  sx, sy, angle)
    lf   = _rotate([(sx-bw//2, sy), (sx-bw//2-10, sy+10),
                    (sx-bw//2, sy-6)],                   sx, sy, angle)
    rf   = _rotate([(sx+bw//2, sy), (sx+bw//2+10, sy+10),
                    (sx+bw//2, sy-6)],                   sx, sy, angle)
    for shape in (body, nose, lf, rf):
        pygame.draw.polygon(surf, ROCKET_COL, shape)

    if throttle > 0.02:
        nozzle = angle + gimbal * MAX_GIMBAL_ANGLE
        flen   = int(throttle * 50)
        for i, col in enumerate(FLAME_COLS):
            fw = max(1, bw // 2 - i * 2)
            tx = sx + np.sin(nozzle) * flen
            ty = sy + np.cos(nozzle) * flen
            px = np.cos(nozzle) * (fw - i * 2)
            py = -np.sin(nozzle) * (fw - i * 2)
            pygame.draw.polygon(surf, col,
                                [(sx - px, sy - py), (sx + px, sy + py), (tx, ty)])


def draw_throttle_bar(surf, font_sm, throttle):
    bx, by, bw, bh = W - 30, 70, 16, 200
    pygame.draw.rect(surf, (40, 40, 40), (bx, by, bw, bh), border_radius=4)
    fill_h = int(throttle * bh)
    if fill_h > 0:
        col = (int(255 * throttle), int(255 * (1 - throttle * 0.5)), 40)
        pygame.draw.rect(surf, col, (bx, by + bh - fill_h, bw, fill_h), border_radius=4)
    pygame.draw.rect(surf, (150, 150, 150), (bx, by, bw, bh), 2, border_radius=4)
    lbl = font_sm.render("THR", True, DIM_COL)
    surf.blit(lbl, (bx - 2, by + bh + 6))


def draw_hud(surf, font_lg, font_sm, state, episode, n_episodes, title, outcome=None):
    lines = [
        title,
        "",
        f"Episode  {episode:2d} / {n_episodes}",
        f"Alt    {state['y']:7.1f} m",
        f"X pos  {state['x']:+7.1f} m",
        f"Vel X  {state['vx']:+7.1f} m/s",
        f"Vel Y  {state['vy']:+7.1f} m/s",
        f"Fuel   {state['fuel']:7.1f} kg",
    ]

    panel_w  = 262
    panel_h  = len(lines) * 20 + 18
    panel    = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill(HUD_BG)
    surf.blit(panel, (8, 8))

    for i, line in enumerate(lines):
        col = TITLE_COL if i == 0 else TEXT_COL
        surf.blit(font_sm.render(line, True, col), (16, 16 + i * 20))

    if outcome:
        col    = LAND_COL if "LAND" in outcome else CRASH_COL
        banner = font_lg.render(outcome, True, col)
        # Semi-transparent backing so banner is readable over any background
        pad    = 12
        bw, bh = banner.get_width() + pad*2, banner.get_height() + pad*2
        bg     = pygame.Surface((bw, bh), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        bx     = W // 2 - bw // 2
        by     = H // 2 - bh // 2
        surf.blit(bg, (bx, by))
        surf.blit(banner, (bx + pad, by + pad))


# ── Policy loaders ────────────────────────────────────────────────────────────

def make_random_policy():
    """Return a callable that samples a random action from the env's action space."""
    def policy(env, _obs):
        return env.action_space.sample()
    return policy


def load_trained_policy(model_path, vec_normalize_path, stage):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    vec_env = make_vec_env(lambda: RocketLandingEnv(stage=stage), n_envs=1)
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False
    model = PPO.load(model_path, env=vec_env)

    def policy(env, obs):
        obs_n = vec_env.normalize_obs(obs.reshape(1, -1))[0]
        action, _ = model.predict(obs_n, deterministic=True)
        return action

    return policy


# ── Recording ─────────────────────────────────────────────────────────────────

BANNER_HOLD = SIM_FPS * 2   # frames the outcome banner stays on screen


def record(stage, n_episodes, output_path, title, model_path=None, vec_normalize_path=None):
    global WORLD_HEIGHT
    cfg          = STAGES[stage]
    WORLD_HEIGHT = cfg["alt"] * 1.1

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Recording → {os.path.basename(output_path)}")

    font_lg = pygame.font.SysFont("consolas", 52, bold=True)
    font_sm = pygame.font.SysFont("consolas", 17)

    if model_path:
        print(f"Loading model: {model_path}")
        policy = load_trained_policy(model_path, vec_normalize_path, stage)
        print("Model loaded.")
    else:
        print("No model — using random (untrained) policy.")
        policy = make_random_policy()

    env   = RocketLandingEnv(stage=stage)
    trail = []
    MAX_TRAIL = 100

    total_frames = 0
    outcomes = []

    print(f"Recording {n_episodes} episodes → {output_path}")
    print(f"Window resolution: {W}×{H}  |  video fps: {SIM_FPS}")

    with imageio.get_writer(
        output_path,
        fps=SIM_FPS,
        macro_block_size=None,      # disable automatic resize to multiples of 16
        pixelformat="yuv420p",      # required for broad player compatibility
        codec="libx264",
        output_params=["-crf", "18"],   # quality: 0=lossless, 51=worst; 18 is near-lossless
    ) as writer:

        def capture():
            frame = pygame.surfarray.array3d(screen)
            writer.append_data(np.ascontiguousarray(frame.transpose(1, 0, 2)))

        for ep in range(1, n_episodes + 1):
            obs, _ = env.reset()
            trail.clear()
            done    = False
            outcome = None

            # ── Episode ──────────────────────────────────────────────────────
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                action = policy(env, obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                sx, sy = world_to_screen(env._state["x"], env._state["y"])
                trail.append((sx, sy))
                if len(trail) > MAX_TRAIL:
                    trail.pop(0)

                draw_background(screen, cfg["pad"], trail, env._state)
                draw_rocket(screen, env._state)
                draw_throttle_bar(screen, font_sm, env._state["throttle"])
                draw_hud(screen, font_lg, font_sm, env._state, ep, n_episodes, title)
                pygame.display.flip()

                capture()
                total_frames += 1

            # ── Outcome ──────────────────────────────────────────────────────
            s        = env._state
            speed_ok = abs(s["vy"]) <= MAX_LANDING_VY and abs(s["vx"]) <= MAX_LANDING_VX
            upright  = abs(s["angle"]) <= MAX_TILT
            on_pad   = abs(s["x"]) <= cfg["pad"]

            if   s["y"] <= 1.0 and speed_ok and upright and on_pad: outcome = "LANDED!"
            elif s["fuel"] <= 0.0:                                   outcome = "Out of fuel"
            elif s["y"] <= 1.0:                                      outcome = "CRASH"
            else:                                                     outcome = "Timeout"

            outcomes.append(outcome)
            print(f"  Ep {ep:2d}/{n_episodes}: {outcome:<16}  "
                  f"vy={s['vy']:+5.1f}  vx={s['vx']:+5.1f}  "
                  f"x={s['x']:+6.1f}m  fuel={s['fuel']:.1f}kg")

            # ── Banner hold ──────────────────────────────────────────────────
            for _ in range(BANNER_HOLD):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                draw_background(screen, cfg["pad"], trail, env._state)
                draw_rocket(screen, env._state)
                draw_throttle_bar(screen, font_sm, env._state["throttle"])
                draw_hud(screen, font_lg, font_sm, env._state, ep, n_episodes,
                         title, outcome=outcome)
                pygame.display.flip()
                capture()
                total_frames += 1

    pygame.quit()

    landed = sum(1 for o in outcomes if o == "LANDED!")
    duration_s = total_frames / SIM_FPS
    print(f"\n{'─'*52}")
    print(f"Saved:    {output_path}")
    print(f"Frames:   {total_frames}  ({duration_s:.1f} s at {SIM_FPS} fps)")
    print(f"Success:  {landed}/{n_episodes} landings")
    print(f"{'─'*52}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record rocket landing demo to MP4")
    parser.add_argument("--stage",         type=int, default=1,
                        help="curriculum stage (default: 1)")
    parser.add_argument("--episodes",      type=int, default=10,
                        help="number of episodes to record (default: 10)")
    parser.add_argument("--output",        type=str, default=None,
                        help="output MP4 path (default: stage<N>_demo.mp4)")
    parser.add_argument("--title",         type=str, default=None,
                        help="HUD title line (default: 'Stage N')")
    parser.add_argument("--model",         type=str, default=None,
                        help="path to best_model WITHOUT .zip extension; "
                             "omit for random/untrained policy")
    parser.add_argument("--vec-normalize", type=str, default=None,
                        help="path to vec_normalize.pkl (required when --model is set)")
    args = parser.parse_args()

    if args.model and not args.vec_normalize:
        parser.error("--vec-normalize is required when --model is given")

    out   = args.output or f"stage{args.stage}_demo.mp4"
    title = args.title  or f"Stage {args.stage}"

    record(
        stage=args.stage,
        n_episodes=args.episodes,
        output_path=out,
        title=title,
        model_path=args.model,
        vec_normalize_path=args.vec_normalize,
    )
