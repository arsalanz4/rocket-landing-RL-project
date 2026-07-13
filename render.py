"""
Pygame renderer for the 2D Rocket Landing environment.

Run modes
---------
  python rocket/render.py                   # watch the trained PPO agent
  python rocket/render.py --heuristic       # watch the simple hand-coded pilot
  python rocket/render.py --human           # you control the throttle (SPACE = full, else idle)

Controls (all modes)
--------------------
  R          restart episode
  Q / Esc    quit
  +/-        speed up / slow down playback
"""

import argparse
import sys
import os

import numpy as np
import pygame

# Allow running from the project root
sys.path.insert(0, os.path.dirname(__file__))
from rocket_env import (
    RocketLandingEnv, FUEL_CAPACITY, STAGES,
    MAX_LANDING_VY, MAX_LANDING_VX,
    MAX_GIMBAL_ANGLE, MAX_TILT,
)

# Load stage-1 config for display scaling (renderer uses stage from the env)
_s1 = STAGES[1]
INIT_ALTITUDE    = _s1["alt"]
LANDING_PAD_HALF = _s1["pad"]

# ── Display constants ─────────────────────────────────────────────────────────
W, H          = 900, 700        # window size in pixels
FPS_DEFAULT   = 30              # target framerate
WORLD_WIDTH   = 300.0           # metres shown horizontally (±150 m)
WORLD_HEIGHT  = INIT_ALTITUDE * 1.05  # metres shown vertically

# Colours (R, G, B)
SKY_TOP       = (10,  10,  40)
SKY_BOT       = (30,  60, 120)
GROUND_COL    = (60,  60,  60)
PAD_COL       = (220, 180,  40)
ROCKET_COL    = (220, 220, 220)
FLAME_COLS    = [(255,200,0), (255,120,0), (200,50,0)]  # inner → outer
TRAIL_COL     = (180, 100,  40, 120)   # with alpha
TEXT_COL      = (230, 230, 230)
HUD_BG        = (0,   0,   0,  160)
CRASH_COL     = (255,  60,  20)
LAND_COL      = (80,  220,  80)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def world_to_screen(x, y):
    """Convert simulation coordinates (metres) to pixel position."""
    px = int(W / 2 + x * (W / WORLD_WIDTH))
    py = int(H   - y * (H / WORLD_HEIGHT))
    return px, py


def metres_to_px(m):
    """Scale a distance in metres to pixels (horizontal scale)."""
    return int(m * (W / WORLD_WIDTH))


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_gradient_sky(surf):
    """Simple vertical gradient from deep navy to mid-blue."""
    for row in range(H):
        t = row / H
        r = int(SKY_TOP[0] * (1-t) + SKY_BOT[0] * t)
        g = int(SKY_TOP[1] * (1-t) + SKY_BOT[1] * t)
        b = int(SKY_TOP[2] * (1-t) + SKY_BOT[2] * t)
        pygame.draw.line(surf, (r, g, b), (0, row), (W, row))


def draw_ground(surf, pad_half=LANDING_PAD_HALF):
    ground_y = world_to_screen(0, 0)[1]
    pygame.draw.rect(surf, GROUND_COL, (0, ground_y, W, H - ground_y))

    # Landing pad (size passed in so it reflects current curriculum stage)
    pad_left  = world_to_screen(-pad_half, 0)[0]
    pad_right = world_to_screen( pad_half, 0)[0]
    pygame.draw.rect(surf, PAD_COL, (pad_left, ground_y - 4, pad_right - pad_left, 4))

    # Pad edge markers
    for mx in (-pad_half, pad_half):
        sx = world_to_screen(mx, 0)[0]
        pygame.draw.rect(surf, PAD_COL, (sx - 3, ground_y - 12, 6, 12))


def _rotate(points, cx, cy, angle_rad):
    """Rotate a list of (x,y) points around (cx,cy) by angle_rad."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    out = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        out.append((cx + dx * cos_a - dy * sin_a,
                    cy + dx * sin_a + dy * cos_a))
    return out


def draw_rocket(surf, sx, sy, throttle, angle_rad=0.0, gimbal=0.0):
    """
    Draw a tilted rocket at screen position (sx, sy).
    angle_rad : body tilt from vertical (positive = right)
    gimbal    : normalised gimbal position [-1, 1]
    """
    bw, bh = 14, 36

    # All points defined with rocket pointing UP (angle=0), pivoted at (sx, sy)
    # sy is the base (engine end) of the rocket.

    # Body corners (bottom-left, top-left, top-right, bottom-right)
    body = [
        (sx - bw//2, sy),
        (sx - bw//2, sy - bh),
        (sx + bw//2, sy - bh),
        (sx + bw//2, sy),
    ]
    nose = [(sx, sy - bh - 16), (sx - bw//2, sy - bh), (sx + bw//2, sy - bh)]

    fin_h = 10
    left_fin  = [(sx - bw//2, sy), (sx - bw//2 - 10, sy + fin_h), (sx - bw//2, sy - 6)]
    right_fin = [(sx + bw//2, sy), (sx + bw//2 + 10, sy + fin_h), (sx + bw//2, sy - 6)]

    # Rotate all shapes around the base centre (sx, sy)
    body      = _rotate(body,      sx, sy, angle_rad)
    nose      = _rotate(nose,      sx, sy, angle_rad)
    left_fin  = _rotate(left_fin,  sx, sy, angle_rad)
    right_fin = _rotate(right_fin, sx, sy, angle_rad)

    pygame.draw.polygon(surf, ROCKET_COL, body)
    pygame.draw.polygon(surf, ROCKET_COL, nose)
    pygame.draw.polygon(surf, ROCKET_COL, left_fin)
    pygame.draw.polygon(surf, ROCKET_COL, right_fin)

    # Flame: nozzle angle = body tilt + gimbal deflection
    if throttle > 0.02:
        nozzle_angle = angle_rad + gimbal * MAX_GIMBAL_ANGLE
        flame_len = int(throttle * 50)
        for i, col in enumerate(FLAME_COLS):
            width = max(1, bw // 2 - i * 2)
            # Flame tip extends downward along the nozzle axis from the base
            tip_x = sx + np.sin(nozzle_angle) * flame_len
            tip_y = sy + np.cos(nozzle_angle) * flame_len
            # Flame base edges, perpendicular to nozzle axis
            perp_x = np.cos(nozzle_angle) * (width - i * 2)
            perp_y = -np.sin(nozzle_angle) * (width - i * 2)
            flame = [
                (sx - perp_x, sy - perp_y),
                (sx + perp_x, sy + perp_y),
                (tip_x, tip_y),
            ]
            pygame.draw.polygon(surf, col, flame)


def draw_altitude_line(surf, font, altitude):
    """Faint horizontal dashed line at current rocket altitude."""
    _, ay = world_to_screen(0, altitude)
    if 0 < ay < H:
        for x in range(0, W, 20):
            pygame.draw.line(surf, (80, 80, 120), (x, ay), (x + 10, ay), 1)
        label = font.render(f"{altitude:.0f} m", True, (100, 100, 160))
        surf.blit(label, (W - 70, ay - 14))


def draw_hud(surf, font_lg, font_sm, state, step, episode, outcome=None):
    """Heads-up display: telemetry panel in the top-left corner."""
    lines = [
        f"Episode  {episode}",
        f"Step     {step}",
        "",
        f"Alt    {state['y']:7.1f} m",
        f"Vel X  {state['vx']:+7.1f} m/s",
        f"Vel Y  {state['vy']:+7.1f} m/s",
        f"Angle  {np.degrees(state['angle']):+7.1f} deg",
        f"AngVel {state['ang_vel']:+7.2f} r/s",
        f"Fuel   {state['fuel']:7.1f} kg  ({100*state['fuel']/FUEL_CAPACITY:.0f}%)",
        f"Throttle {state['throttle']*100:5.1f}%",
        f"Gimbal   {state['gimbal']*100:+5.1f}%",
    ]

    # Background panel
    panel_w, panel_h = 230, len(lines) * 20 + 16
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill(HUD_BG)
    surf.blit(panel, (8, 8))

    for i, line in enumerate(lines):
        label = font_sm.render(line, True, TEXT_COL)
        surf.blit(label, (16, 16 + i * 20))

    # Big outcome banner
    if outcome:
        col   = LAND_COL if "LAND" in outcome else CRASH_COL
        banner = font_lg.render(outcome, True, col)
        rx = W // 2 - banner.get_width() // 2
        ry = H // 2 - banner.get_height() // 2
        surf.blit(banner, (rx, ry))


def draw_throttle_bar(surf, font_sm, throttle):
    """Vertical throttle gauge on the right edge."""
    bx, by, bw, bh = W - 30, 60, 16, 200
    pygame.draw.rect(surf, (50, 50, 50), (bx, by, bw, bh), border_radius=4)
    fill_h = int(throttle * bh)
    fill_col = (
        int(255 * throttle),
        int(255 * (1 - throttle * 0.5)),
        40,
    )
    if fill_h > 0:
        pygame.draw.rect(surf, fill_col,
                         (bx, by + bh - fill_h, bw, fill_h), border_radius=4)
    pygame.draw.rect(surf, (180, 180, 180), (bx, by, bw, bh), 2, border_radius=4)
    label = font_sm.render("THR", True, TEXT_COL)
    surf.blit(label, (bx - 2, by + bh + 6))


# ── Pilots ────────────────────────────────────────────────────────────────────

def heuristic_action(obs):
    """Simple hand-coded pilot: brake hard when fast or close to ground; PD gimbal stabiliser."""
    alt, vy, angle, ang_vel = obs[1], obs[3], obs[4], obs[5]
    throttle = 1.0 if (vy < -15 or alt < 100) else 0.3
    gimbal = float(np.clip(angle * 3.0 + ang_vel * 1.0, -1.0, 1.0))
    return np.array([throttle, gimbal], dtype=np.float32)


def load_ppo_agent():
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    model_path = "rocket/best_model.zip"
    norm_path  = "rocket/vec_normalize.pkl"

    if not os.path.exists(model_path):
        print("No trained model found at rocket/best_model.zip")
        print("Run: python rocket/train_ppo.py")
        sys.exit(1)

    vec_env = make_vec_env(lambda: RocketLandingEnv(), n_envs=1)
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training   = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)
    return model, vec_env


# ── Main render loop ──────────────────────────────────────────────────────────

def run(mode: str):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Rocket Landing")
    clock  = pygame.time.Clock()

    font_lg = pygame.font.SysFont("consolas", 52, bold=True)
    font_sm = pygame.font.SysFont("consolas", 17)

    # Pre-render the static sky so we don't redo the gradient every frame
    sky_surf = pygame.Surface((W, H))
    draw_gradient_sky(sky_surf)

    # Trail: list of (sx, sy) pixel positions
    trail = []
    MAX_TRAIL = 120

    # Set up agent
    ppo_model, ppo_vec_env = None, None
    if mode == "ppo":
        ppo_model, ppo_vec_env = load_ppo_agent()

    env     = RocketLandingEnv()
    obs, _  = env.reset()
    ppo_obs = ppo_vec_env.reset() if ppo_model else None

    episode  = 1
    step     = 0
    outcome  = None          # shown as banner when episode ends
    outcome_timer = 0        # frames to keep the banner visible
    fps      = FPS_DEFAULT
    done     = False

    running = True
    while running:
        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, _  = env.reset()
                    ppo_obs = ppo_vec_env.reset() if ppo_model else None
                    trail.clear()
                    episode += 1
                    step     = 0
                    outcome  = None
                    done     = False
                elif event.key == pygame.K_EQUALS:   # +
                    fps = min(fps + 10, 120)
                elif event.key == pygame.K_MINUS:    # -
                    fps = max(fps - 10, 5)

        # ── Step simulation (skip if episode ended) ───────────────────────────
        if not done:
            if mode == "ppo" and ppo_model:
                action, _ = ppo_model.predict(ppo_obs, deterministic=True)
                ppo_obs, _, ppo_done, _ = ppo_vec_env.step(action)
                # Step the raw env with the same action for rendering
                obs, _, terminated, truncated, info = env.step(action[0])
                done = bool(ppo_done[0])
            elif mode == "heuristic":
                action = heuristic_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            else:  # human
                keys = pygame.key.get_pressed()
                throttle = 1.0 if keys[pygame.K_SPACE] else 0.0
                gimbal   = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (1.0 if keys[pygame.K_LEFT] else 0.0)
                action = np.array([throttle, gimbal], dtype=np.float32)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            step += 1

            # Append rocket pixel position to trail
            sx, sy = world_to_screen(env._state["x"], env._state["y"])
            trail.append((sx, sy))
            if len(trail) > MAX_TRAIL:
                trail.pop(0)

            if done:
                s = env._state
                speed_ok = abs(s["vy"]) <= MAX_LANDING_VY and abs(s["vx"]) <= MAX_LANDING_VX
                upright  = abs(s["angle"]) <= MAX_TILT
                pad_half = STAGES[env.stage]["pad"]
                on_pad   = abs(s["x"]) <= pad_half
                if s["y"] <= 1.0 and speed_ok and upright and on_pad:
                    outcome = "LANDED!"
                elif s["y"] <= 1.0 and speed_ok and upright:
                    outcome = "Soft (off pad)"
                elif s["y"] <= 1.0:
                    outcome = "CRASH"
                elif s["fuel"] <= 0:
                    outcome = "Out of fuel"
                else:
                    outcome = "Timeout"
                outcome_timer = fps * 2   # show for 2 seconds

        else:
            # Countdown banner, then auto-reset
            outcome_timer -= 1
            if outcome_timer <= 0:
                obs, _  = env.reset()
                ppo_obs = ppo_vec_env.reset() if ppo_model else None
                trail.clear()
                episode += 1
                step     = 0
                outcome  = None
                done     = False

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.blit(sky_surf, (0, 0))
        draw_ground(screen, pad_half=STAGES[env.stage]["pad"])

        # Exhaust trail (fades by alpha — draw oldest first)
        for i, (tx, ty) in enumerate(trail):
            alpha = int(200 * i / max(len(trail), 1))
            radius = max(1, 3 - (len(trail) - i) // 30)
            trail_surf = pygame.Surface((radius*2+1, radius*2+1), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (*TRAIL_COL[:3], alpha), (radius, radius), radius)
            screen.blit(trail_surf, (tx - radius, ty - radius))

        draw_altitude_line(screen, font_sm, env._state["y"])

        sx, sy = world_to_screen(env._state["x"], env._state["y"])
        draw_rocket(screen, sx, sy, env._state["throttle"],
                    angle_rad=env._state["angle"], gimbal=env._state["gimbal"])
        draw_throttle_bar(screen, font_sm, env._state["throttle"])
        draw_hud(screen, font_lg, font_sm, env._state, step, episode,
                 outcome=outcome if done else None)

        # FPS counter bottom-right
        fps_label = font_sm.render(f"{clock.get_fps():.0f} fps  ({fps} target)", True, (120,120,120))
        screen.blit(fps_label, (W - fps_label.get_width() - 10, H - 24))

        # Controls hint bottom-left
        hints = "R=restart  +/-=speed  Q=quit"
        if mode == "human":
            hints = "SPACE=throttle  LEFT/RIGHT=gimbal  " + hints
        hint_label = font_sm.render(hints, True, (120, 120, 120))
        screen.blit(hint_label, (10, H - 24))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--heuristic", action="store_true", help="run the hand-coded pilot")
    group.add_argument("--human",     action="store_true", help="control throttle yourself (SPACE)")
    args = parser.parse_args()

    if args.heuristic:
        run("heuristic")
    elif args.human:
        run("human")
    else:
        run("ppo")
