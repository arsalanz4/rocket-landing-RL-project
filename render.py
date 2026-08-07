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
    MAX_GIMBAL_ANGLE, MAX_TILT, MAX_WIND,
)

# ── Display constants ─────────────────────────────────────────────────────────
W, H          = 900, 700        # window size in pixels
FPS_DEFAULT   = 30              # target framerate
WORLD_WIDTH   = 300.0           # metres shown horizontally (±150 m)
WORLD_HEIGHT  = STAGES[1]["alt"] * 1.05   # set dynamically in run(); default = stage 1

# Colours (R, G, B)
SKY_TOP       = (6,    8,  26)
SKY_BOT       = (30,  60, 120)
GROUND_COL    = (60,  60,  60)
PAD_COL       = (220, 180,  40)
PAD_DARK      = (40,   40,  40)

ROCKET_COL    = (225, 225, 230)
ROCKET_SHADE  = (145, 145, 155)   # shadow-side shading for a 3-D cylinder look
ROCKET_DARK   = (55,   55,  65)   # nozzle / window bezel
WINDOW_COL    = (120, 195, 235)   # cockpit window
FLAME_COLS    = [(170, 45, 10), (255, 140, 30), (255, 235, 170)]  # outer (dim red) → inner (hot core)
FLAME_GLOW    = (255, 150, 40, 60)
TRAIL_HEAD    = (255, 150, 40)    # warm colour at the rocket end of the trail
TRAIL_COL     = (180, 100,  40, 120)   # with alpha (tail end)
TEXT_COL      = (230, 230, 230)
HUD_BG        = (0,   0,   0,  160)
CRASH_COL     = (255,  60,  20)
LAND_COL      = (80,  220,  80)
WIND_COL      = (80,  200, 255)   # cyan for wind indicator

# Background dressing
STAR_SEED         = 42
MOON_COL          = (200, 198, 190)
MOON_SHADE_COL    = (90,   90,  95)
RINGED_COL        = (210, 180, 130)   # distant ringed planet, Saturn-like
RINGED_SHADE_COL  = (110,  90,  60)
SHOOTING_STAR_COL = (255, 255, 255)


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


def draw_stars(surf, seed=STAR_SEED, count=160):
    """Scatter faint stars into the upper sky, dimmer against the brighter horizon."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        x = int(rng.uniform(0, W))
        y = int(rng.uniform(0, H * 0.8))
        t = y / H
        bg = (
            SKY_TOP[0] * (1-t) + SKY_BOT[0] * t,
            SKY_TOP[1] * (1-t) + SKY_BOT[1] * t,
            SKY_TOP[2] * (1-t) + SKY_BOT[2] * t,
        )
        brightness = rng.uniform(0.3, 1.0)
        col = tuple(int(bg[i] + (255 - bg[i]) * brightness) for i in range(3))
        if rng.random() < 0.12:
            pygame.draw.circle(surf, col, (x, y), 2)
        else:
            surf.set_at((x, y), col)


def draw_planet(surf, cx, cy, r, base_col, shade_col, ring=False):
    """A shaded sphere (optionally with a flattened ring) for background dressing."""
    pad = int(r * 1.6)
    size = r * 2 + pad * 2
    pc = size // 2
    planet = pygame.Surface((size, size), pygame.SRCALPHA)

    if ring:
        ring_rect = pygame.Rect(0, 0, int(r * 2.7), int(r * 0.8))
        ring_rect.center = (pc, pc)
        pygame.draw.ellipse(planet, (*shade_col, 220), ring_rect, width=max(2, r // 5))

    pygame.draw.circle(planet, (*base_col, 255), (pc, pc), r)

    # Terminator shading: darken one side, respecting the sphere's alpha mask
    shade = pygame.Surface((size, size), pygame.SRCALPHA)
    shade.fill((255, 255, 255, 255))
    pygame.draw.circle(shade, (*shade_col, 255), (pc + int(r * 0.55), pc), r)
    planet.blit(shade, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    if ring:
        front_rect = pygame.Rect(0, 0, int(r * 2.7), int(r * 0.35))
        front_rect.center = (pc, pc + int(r * 0.25))
        pygame.draw.ellipse(planet, (*shade_col, 200), front_rect, width=max(1, r // 6))

    surf.blit(planet, (cx - pc, cy - pc))


def draw_space_dressing(surf):
    """Stars + a couple of distant planets, pre-baked once onto the static sky."""
    draw_stars(surf)
    draw_planet(surf, int(W * 0.82), int(H * 0.16), 20, MOON_COL, MOON_SHADE_COL)
    draw_planet(surf, int(W * 0.66), int(H * 0.32), 11, RINGED_COL, RINGED_SHADE_COL, ring=True)


def draw_shooting_star(surf, x, y, dx, dy, intensity):
    """A bright head with a short fading tail streaking across the sky, drawn additively."""
    for i in range(9):
        px, py = x - dx * i * 5, y - dy * i * 5
        if not (0 <= px < W and 0 <= py < H):
            continue
        alpha = int(max(0, 235 - i * 28) * intensity)
        if alpha <= 0:
            continue
        r = max(1, 2 - i // 4)
        patch = pygame.Surface((r * 2 + 1, r * 2 + 1), pygame.SRCALPHA)
        pygame.draw.circle(patch, (*SHOOTING_STAR_COL, alpha), (r, r), r)
        surf.blit(patch, (int(px - r), int(py - r)), special_flags=pygame.BLEND_RGBA_ADD)


def draw_ground(surf, pad_half=40.0, font_sm=None):
    ground_y = world_to_screen(0, 0)[1]
    pygame.draw.rect(surf, GROUND_COL, (0, ground_y, W, H - ground_y))

    # Landing pad (size passed in so it reflects current curriculum stage)
    pad_left  = world_to_screen(-pad_half, 0)[0]
    pad_right = world_to_screen( pad_half, 0)[0]
    pad_cx    = (pad_left + pad_right) // 2

    # Hazard-stripe deck along the top of the pad
    stripe_h, stripe_w = 6, 14
    x, toggle = pad_left, True
    while x < pad_right:
        seg_w = min(stripe_w, pad_right - x)
        pygame.draw.rect(surf, PAD_COL if toggle else PAD_DARK,
                         (x, ground_y - stripe_h, seg_w, stripe_h))
        x += seg_w
        toggle = not toggle

    # Bullseye target rings marking the precise touchdown point
    for i, r in enumerate((15, 9)):
        pygame.draw.circle(surf, PAD_COL, (pad_cx, ground_y - stripe_h - 1), r, width=2)
    pygame.draw.circle(surf, PAD_COL, (pad_cx, ground_y - stripe_h - 1), 2)

    # Pad edge markers (corner posts)
    for mx in (-pad_half, pad_half):
        sx = world_to_screen(mx, 0)[0]
        pygame.draw.rect(surf, PAD_COL, (sx - 3, ground_y - stripe_h - 12, 6, 12))

    if font_sm is not None:
        label = font_sm.render("LANDING ZONE", True, PAD_COL)
        surf.blit(label, (pad_cx - label.get_width() // 2, ground_y - stripe_h - 30))


def _rotate(points, cx, cy, angle_rad):
    """Rotate a list of (x,y) points around (cx,cy) by angle_rad."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    out = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        out.append((cx + dx * cos_a - dy * sin_a,
                    cy + dx * sin_a + dy * cos_a))
    return out


def _flame_polygon(bx, by, nozzle_angle, length, base_width, wobble_phase, segments=6):
    """An organically tapered flame silhouette (not a straight-edged triangle)."""
    fwd  = np.array([np.sin(nozzle_angle),  np.cos(nozzle_angle)])
    perp = np.array([np.cos(nozzle_angle), -np.sin(nozzle_angle)])
    base = np.array([bx, by])
    left_pts, right_pts = [], []
    for i in range(segments + 1):
        s = i / segments
        w = base_width * (1 - s) ** 0.8 * (1 + 0.12 * np.sin(s * 9 + wobble_phase))
        centre = base + fwd * length * s
        left_pts.append(tuple(centre - perp * w))
        right_pts.append(tuple(centre + perp * w))
    tip = base + fwd * length
    return left_pts + [tuple(tip)] + right_pts[::-1]


def _draw_flame(surf, bx, by, nozzle_angle, flame_len, base_width):
    """A layered, flickering exhaust plume with a hot core and a soft glow."""
    if flame_len < 2:
        return
    phase = pygame.time.get_ticks() * 0.02

    # Soft additive glow bloom near the nozzle
    glow_r = max(3, int(flame_len * 0.26))
    glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
    pygame.draw.circle(glow_surf, FLAME_GLOW, (glow_r, glow_r), glow_r)
    mid_x = bx + np.sin(nozzle_angle) * flame_len * 0.25
    mid_y = by + np.cos(nozzle_angle) * flame_len * 0.25
    surf.blit(glow_surf, (int(mid_x - glow_r), int(mid_y - glow_r)),
              special_flags=pygame.BLEND_RGBA_ADD)

    # Nested layers: dim outer plume -> mid orange -> bright core, each with
    # its own flicker phase so they wobble independently (more organic).
    layers = [
        (flame_len * 1.00, base_width * 1.00, FLAME_COLS[0], phase),
        (flame_len * 0.72, base_width * 0.62, FLAME_COLS[1], phase * 1.3 + 2.0),
        (flame_len * 0.45, base_width * 0.32, FLAME_COLS[2], phase * 1.6 + 4.0),
    ]
    for length, width, col, wobble_phase in layers:
        poly = _flame_polygon(bx, by, nozzle_angle, length, width, wobble_phase)
        pygame.draw.polygon(surf, col, poly)


def draw_rocket(surf, sx, sy, throttle, angle_rad=0.0, gimbal=0.0):
    """
    Draw a tilted rocket at screen position (sx, sy).
    angle_rad : body tilt from vertical (positive = right)
    gimbal    : normalised gimbal position [-1, 1]
    """
    bw, bh = 20, 50

    # All points defined with rocket pointing UP (angle=0), pivoted at (sx, sy)
    # sy is the base (engine end) of the rocket.

    # Body corners (bottom-left, top-left, top-right, bottom-right)
    body = [
        (sx - bw//2, sy),
        (sx - bw//2, sy - bh),
        (sx + bw//2, sy - bh),
        (sx + bw//2, sy),
    ]
    # Shadow strip along the trailing edge, for a subtle 3-D cylinder look
    shade = [
        (sx + bw//6, sy),
        (sx + bw//6, sy - bh),
        (sx + bw//2, sy - bh),
        (sx + bw//2, sy),
    ]
    nose       = [(sx, sy - bh - 22), (sx - bw//2, sy - bh), (sx + bw//2, sy - bh)]
    nose_shade = [(sx, sy - bh - 22), (sx + bw//6, sy - bh), (sx + bw//2, sy - bh)]

    fin_h = 14
    left_fin  = [(sx - bw//2, sy), (sx - bw//2 - 14, sy + fin_h), (sx - bw//2, sy - 8)]
    right_fin = [(sx + bw//2, sy), (sx + bw//2 + 14, sy + fin_h), (sx + bw//2, sy - 8)]

    # Engine nozzle stub at the base, between the fins
    nozzle = [
        (sx - bw//2 + 3, sy),
        (sx - bw//3,     sy + 8),
        (sx + bw//3,     sy + 8),
        (sx + bw//2 - 3, sy),
    ]

    window_pt = (sx, sy - bh * 0.7)   # cockpit window, upper third of the body

    # Rotate everything around the base centre (sx, sy)
    body      = _rotate(body,      sx, sy, angle_rad)
    shade     = _rotate(shade,     sx, sy, angle_rad)
    nose      = _rotate(nose,      sx, sy, angle_rad)
    nose_shade= _rotate(nose_shade,sx, sy, angle_rad)
    left_fin  = _rotate(left_fin,  sx, sy, angle_rad)
    right_fin = _rotate(right_fin, sx, sy, angle_rad)
    nozzle    = _rotate(nozzle,    sx, sy, angle_rad)
    wx, wy    = _rotate([window_pt], sx, sy, angle_rad)[0]

    # Flame drawn first so it appears to emit from behind the nozzle
    if throttle > 0.02:
        nozzle_angle = angle_rad + gimbal * MAX_GIMBAL_ANGLE
        flicker = 1.0 + 0.07 * np.sin(pygame.time.get_ticks() * 0.035) \
                       + 0.03 * np.sin(pygame.time.get_ticks() * 0.11)
        flame_len = throttle * 72 * flicker
        _draw_flame(surf, sx, sy, nozzle_angle, flame_len, bw * 0.42)

    pygame.draw.polygon(surf, ROCKET_COL, body)
    pygame.draw.polygon(surf, ROCKET_COL, nose)
    pygame.draw.polygon(surf, ROCKET_SHADE, shade)
    pygame.draw.polygon(surf, ROCKET_SHADE, nose_shade)
    pygame.draw.polygon(surf, ROCKET_COL, left_fin)
    pygame.draw.polygon(surf, ROCKET_COL, right_fin)
    pygame.draw.polygon(surf, ROCKET_DARK, nozzle)
    pygame.draw.circle(surf, ROCKET_DARK, (int(wx), int(wy)), 7)
    pygame.draw.circle(surf, WINDOW_COL, (int(wx), int(wy)), 4)


def draw_altitude_line(surf, font, altitude):
    """Faint horizontal dashed line at current rocket altitude."""
    _, ay = world_to_screen(0, altitude)
    if 0 < ay < H:
        for x in range(0, W, 20):
            pygame.draw.line(surf, (80, 80, 120), (x, ay), (x + 10, ay), 1)
        label = font.render(f"{altitude:.0f} m", True, (100, 100, 160))
        surf.blit(label, (W - 70, ay - 14))


def draw_hud(surf, font_lg, font_sm, state, step, episode, stage, outcome=None):
    """Heads-up display: telemetry panel in the top-left corner."""
    wind_f = state.get("wind_force", 0.0)
    wind_a = state.get("wind_active", False)
    wind_tag = " GUST" if wind_a else "     "

    lines = [
        f"Stage    {stage}",
        f"Episode  {episode}",
        f"Step     {step}",
        "",
        f"Alt    {state['y']:7.1f} m",
        f"Vel X  {state['vx']:+7.1f} m/s",
        f"Vel Y  {state['vy']:+7.1f} m/s",
        f"Angle  {np.degrees(state['angle']):+7.1f} deg",
        f"AngVel {state['ang_vel']:+7.2f} r/s",
        f"Fuel   {state['fuel']:7.1f} kg",
        f"Throttle {state['throttle']*100:5.1f}%",
        f"Gimbal   {state['gimbal']*100:+5.1f}%",
    ]
    wind_idx = len(lines)
    lines.append(f"Wind  {wind_f:+7.1f} m/s²{wind_tag}")
    lines.append("")
    outcome_idx = len(lines)
    lines.append(f"Outcome  {outcome if outcome else '--'}")

    # Background panel
    panel_w, panel_h = 240, len(lines) * 20 + 16
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill(HUD_BG)
    surf.blit(panel, (8, 8))

    for i, line in enumerate(lines):
        if i == wind_idx and wind_a:
            col = WIND_COL
        elif i == outcome_idx and outcome:
            col = LAND_COL if "LAND" in outcome else CRASH_COL
        else:
            col = TEXT_COL
        label = font_sm.render(line, True, col)
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


def draw_wind_indicator(surf, font_sm, wind_force, wind_active):
    """
    Horizontal wind bar centred at the top of the screen.
    The bar extends left/right proportional to wind magnitude.
    Cyan = active gust; dim blue = no wind.
    """
    cx, cy   = W // 2, 22
    max_px   = 90       # full-bar pixel half-width at MAX_WIND
    bar_half = 100      # background bar half-width

    # Background track
    pygame.draw.rect(surf, (30, 30, 50),
                     (cx - bar_half, cy - 4, bar_half * 2, 8), border_radius=4)
    # Centre tick
    pygame.draw.line(surf, (80, 80, 110), (cx, cy - 8), (cx, cy + 8), 1)

    fill_px = int(wind_force / MAX_WIND * max_px)
    if abs(fill_px) >= 1:
        col = WIND_COL if wind_active else (50, 120, 180)
        if fill_px > 0:
            pygame.draw.rect(surf, col, (cx, cy - 4, fill_px, 8), border_radius=4)
            # Arrowhead pointing right
            pygame.draw.polygon(surf, col, [
                (cx + fill_px + 9, cy),
                (cx + fill_px,     cy - 6),
                (cx + fill_px,     cy + 6),
            ])
        else:
            pygame.draw.rect(surf, col, (cx + fill_px, cy - 4, -fill_px, 8), border_radius=4)
            # Arrowhead pointing left
            pygame.draw.polygon(surf, col, [
                (cx + fill_px - 9, cy),
                (cx + fill_px,     cy - 6),
                (cx + fill_px,     cy + 6),
            ])

    if abs(wind_force) > 0.05:
        gust_str  = "GUST " if wind_active else ""
        label_str = f"{gust_str}WIND {wind_force:+.1f} m/s²"
        col       = WIND_COL if wind_active else (80, 130, 190)
    else:
        label_str = "no wind"
        col       = (60, 60, 90)

    label = font_sm.render(label_str, True, col)
    surf.blit(label, (cx - label.get_width() // 2, cy + 12))


# ── Pilots ────────────────────────────────────────────────────────────────────

def heuristic_action(obs):
    """Simple hand-coded pilot: brake hard when fast or close to ground; PD gimbal stabiliser."""
    alt, vy, angle, ang_vel = obs[1], obs[3], obs[4], obs[5]
    throttle = 1.0 if (vy < -15 or alt < 100) else 0.3
    gimbal = float(np.clip(angle * 3.0 + ang_vel * 1.0, -1.0, 1.0))
    return np.array([throttle, gimbal], dtype=np.float32)


def read_current_stage() -> int:
    """Read the stage that training last saved to current_stage.txt."""
    stage_file = "rocket/current_stage.txt"
    if os.path.exists(stage_file):
        with open(stage_file, encoding="utf-8-sig") as f:
            content = f.read().strip()
            if content:
                return int(content)
    return 1


def load_ppo_agent(stage: int = 1):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    model_path = "rocket/best_model.zip"
    norm_path  = "rocket/vec_normalize.pkl"

    if not os.path.exists(model_path):
        print("No trained model found at rocket/best_model.zip")
        print("Run: python rocket/train_ppo.py")
        sys.exit(1)

    vec_env = make_vec_env(lambda: RocketLandingEnv(stage=stage), n_envs=1)
    vec_env = VecNormalize.load(norm_path, vec_env)
    vec_env.training    = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)
    return model, vec_env


# ── Main render loop ──────────────────────────────────────────────────────────

def run(mode: str, stage: int = 1):
    global WORLD_HEIGHT
    WORLD_HEIGHT = STAGES[stage]["alt"] * 1.1   # scale viewport to current stage altitude

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Rocket Landing  [stage {stage}]")
    clock  = pygame.time.Clock()

    font_lg = pygame.font.SysFont("consolas", 52, bold=True)
    font_sm = pygame.font.SysFont("consolas", 17)

    # Pre-render the static sky so we don't redo the gradient every frame
    sky_surf = pygame.Surface((W, H))
    draw_gradient_sky(sky_surf)
    draw_space_dressing(sky_surf)

    # Trail: list of (sx, sy) pixel positions
    trail = []
    MAX_TRAIL = 120

    # Shooting star: occasional animated streak in the background
    star_rng = np.random.default_rng()
    shooting_star = None
    shooting_star_cooldown = int(star_rng.uniform(60, 200))

    # Set up agent
    ppo_model, ppo_vec_env = None, None
    if mode == "ppo":
        ppo_model, ppo_vec_env = load_ppo_agent(stage=stage)

    env     = RocketLandingEnv(stage=stage)
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

        # Shooting star: spawn occasionally, animate a brief streak, then clear
        if shooting_star is None:
            shooting_star_cooldown -= 1
            if shooting_star_cooldown <= 0:
                lo, hi = [(20, 55), (125, 160)][star_rng.integers(0, 2)]
                angle = np.radians(star_rng.uniform(lo, hi))
                shooting_star = {
                    "x": star_rng.uniform(0, W), "y": star_rng.uniform(0, H * 0.2),
                    "dx": np.sin(angle), "dy": np.cos(angle),
                    "speed": star_rng.uniform(7, 11),
                    "age": 0, "life": int(star_rng.uniform(16, 26)),
                }
        else:
            s = shooting_star
            s["age"] += 1
            s["x"] += s["dx"] * s["speed"]
            s["y"] += s["dy"] * s["speed"]
            t = s["age"] / s["life"]
            intensity = min(1.0, t * 5, (1 - t) * 5)
            draw_shooting_star(screen, s["x"], s["y"], s["dx"], s["dy"], intensity)
            if s["age"] >= s["life"] or s["y"] > H * 0.6:
                shooting_star = None
                shooting_star_cooldown = int(star_rng.uniform(150, 450))

        draw_ground(screen, pad_half=STAGES[env.stage]["pad"], font_sm=font_sm)

        # Exhaust trail: fades by alpha and warms in colour toward the rocket
        n = max(len(trail) - 1, 1)
        for i, (tx, ty) in enumerate(trail):
            t = i / n
            alpha = int(200 * t)
            radius = max(1, 3 - (len(trail) - i) // 30)
            r = int(TRAIL_COL[0] * (1 - t) + TRAIL_HEAD[0] * t)
            g = int(TRAIL_COL[1] * (1 - t) + TRAIL_HEAD[1] * t)
            b = int(TRAIL_COL[2] * (1 - t) + TRAIL_HEAD[2] * t)
            trail_surf = pygame.Surface((radius*2+1, radius*2+1), pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, (r, g, b, alpha), (radius, radius), radius)
            screen.blit(trail_surf, (tx - radius, ty - radius))

        draw_altitude_line(screen, font_sm, env._state["y"])

        sx, sy = world_to_screen(env._state["x"], env._state["y"])
        draw_rocket(screen, sx, sy, env._state["throttle"],
                    angle_rad=env._state["angle"], gimbal=env._state["gimbal"])
        draw_throttle_bar(screen, font_sm, env._state["throttle"])
        draw_wind_indicator(screen, font_sm,
                            env._state.get("wind_force", 0.0),
                            env._state.get("wind_active", False))
        draw_hud(screen, font_lg, font_sm, env._state, step, episode, env.stage,
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
    parser.add_argument("--stage", type=int, default=None,
                        help="curriculum stage to render (default: read from current_stage.txt)")
    args = parser.parse_args()

    render_stage = args.stage if args.stage is not None else read_current_stage()

    if args.heuristic:
        run("heuristic", stage=render_stage)
    elif args.human:
        run("human", stage=render_stage)
    else:
        run("ppo", stage=render_stage)
