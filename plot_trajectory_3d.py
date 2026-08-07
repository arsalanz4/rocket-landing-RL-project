"""
3D scientific-visualisation-style animation of the trained PPO rocket landing agent.

Loads best_model.zip + vec_normalize.pkl, runs a handful of deterministic
evaluation episodes, logs the full physical trajectory of each, and renders
a slowly-orbiting 3D Matplotlib animation: horizontal position x altitude x
"episode lane", with the flight path coloured by throttle, a short vector at
the rocket showing thrust direction (gimbal), a marked landing pad per lane,
and a dark background dressed with a starfield and two background planets.

Usage
-----
  python rocket/plot_trajectory_3d.py                       # 3 episodes, current_stage.txt
  python rocket/plot_trajectory_3d.py --stage 5 --episodes 4
  python rocket/plot_trajectory_3d.py --out my_clip.mp4 --fps 30 --dpi 150
  python rocket/plot_trajectory_3d.py --seeds 2,4,20         # specific seeds instead of a range
  python rocket/plot_trajectory_3d.py --model other.zip --norm other_vecnorm.pkl
  python rocket/plot_trajectory_3d.py --seeds 4 --zoom 12    # single episode -> zoomed close-up
                                                                # 3D rocket model + chase camera

Requires ffmpeg for MP4 export. If no system ffmpeg is found, falls back to
the bundled binary from the `imageio-ffmpeg` package if installed.

Note: --model/--norm must match the observation space the live rocket_env.py
in this repo produces (dimension and stage-config scale). A checkpoint saved
against a differently-shaped or differently-scaled observation space (e.g. an
older commit's env, or a very different curriculum stage) will either fail to
load (shape mismatch) or behave unreliably even if it loads, since
VecNormalize's normalization statistics are calibrated to a specific
observation distribution -- see collect_episode()'s docstring-adjacent
comment in load_agent() for why this script uses a single directly-stepped
env rather than a second VecNormalize-wrapped one.
"""

import argparse
import os
import shutil
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

sys.path.insert(0, os.path.dirname(__file__))
from rocket_env import (
    RocketLandingEnv, STAGES, DT,
    MAX_LANDING_VY, MAX_LANDING_VX, MAX_TILT, MAX_GIMBAL_ANGLE,
)

DEFAULT_MODEL_PATH = "rocket/best_model.zip"
DEFAULT_NORM_PATH  = "rocket/vec_normalize.pkl"

# ── Palette (dark, "scientific" rather than game-like) ─────────────────────
BG_COL        = "#05060f"
PANE_COL      = (0.04, 0.05, 0.10, 1.0)
GRID_COL      = (1, 1, 1, 0.06)
AXIS_COL      = (0.55, 0.58, 0.68, 0.9)
PAD_COL       = "#e0b428"
PAD_FACE      = (0.88, 0.71, 0.16, 0.28)
ROCKET_COL    = "#f2f2f5"
THRUST_COL    = "#ff8a3d"
LAND_COL      = "#4fdc7a"
CRASH_COL     = "#ff4d4d"
MOON_COL      = "#c9c7c0"
PLANET_COL    = "#d59a5a"
STAR_COL      = "#ffffff"
THROTTLE_CMAP = "plasma"

# Rocket model colours -- matches rocket/render.py's pygame palette, just
# expressed as 0-1 floats instead of 0-255 ints, so the 3D model and the 2D
# HUD renderer read as the same vehicle.
ROCKET_HULL_COL  = (225/255, 225/255, 230/255)
ROCKET_SHADE_COL = (145/255, 145/255, 155/255)
ROCKET_DARK_COL  = (55/255,  55/255,  65/255)
ROCKET_WINDOW_COL = (120/255, 195/255, 235/255)
ROCKET_FLAME_COLS = [(210/255, 65/255, 15/255), (255/255, 150/255, 35/255), (255/255, 235/255, 170/255)]


# ── Agent loading & rollout ─────────────────────────────────────────────────

def read_current_stage() -> int:
    stage_file = "rocket/current_stage.txt"
    if os.path.exists(stage_file):
        with open(stage_file, encoding="utf-8-sig") as f:
            content = f.read().strip()
            if content:
                return int(content)
    return 1


def load_agent(stage: int, model_path: str = DEFAULT_MODEL_PATH, norm_path: str = DEFAULT_NORM_PATH):
    """Load the trained PPO policy and its VecNormalize observation statistics."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        print("Run: python rocket/train_ppo.py")
        sys.exit(1)

    # Only used as a structural carrier for VecNormalize's saved obs_rms stats
    # -- never stepped. The actual rollout uses a single raw env directly so
    # the logged trajectory is exactly what the policy acted on, with no risk
    # of a second, independently-seeded env drifting out of sync with it.
    dummy_vec = make_vec_env(lambda: RocketLandingEnv(stage=stage), n_envs=1)
    vec_normalize = VecNormalize.load(norm_path, dummy_vec)
    vec_normalize.training = False
    vec_normalize.norm_reward = False

    model = PPO.load(model_path)
    return model, vec_normalize


def outcome_color(outcome):
    if outcome == "LANDED":
        return LAND_COL
    if outcome == "CRASH":
        return CRASH_COL
    if outcome == "OUT OF FUEL":
        return "#e8a23d"   # ran dry while still controlled -- not a crash
    if "SOFT" in outcome:
        return "#e8c34a"   # soft touchdown, off pad
    return "#9aa0b0"        # TIMEOUT / other -- neutral


def classify_outcome(state, stage):
    cfg = STAGES[stage]
    speed_ok = abs(state["vy"]) <= MAX_LANDING_VY and abs(state["vx"]) <= MAX_LANDING_VX
    upright  = abs(state["angle"]) <= MAX_TILT
    on_pad   = abs(state["x"]) <= cfg["pad"]
    if state["y"] <= 1.0 and speed_ok and upright and on_pad:
        return "LANDED"
    if state["y"] <= 1.0 and speed_ok and upright:
        return "SOFT (off pad)"
    if state["y"] <= 1.0:
        return "CRASH"
    if state["fuel"] <= 0.0:
        return "OUT OF FUEL"
    return "TIMEOUT"


def collect_episode(model, vec_normalize, stage, seed, max_steps):
    """Run one deterministic episode on a raw env, logging the true physical state."""
    env = RocketLandingEnv(stage=stage)
    obs, _ = env.reset(seed=seed)

    frames = []
    outcome = None
    for step_i in range(max_steps):
        norm_obs = vec_normalize.normalize_obs(obs[np.newaxis, :].astype(np.float32))
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action[0])

        s = env._state
        frames.append(dict(
            t=step_i * DT, x=s["x"], y=s["y"], vx=s["vx"], vy=s["vy"],
            angle=s["angle"], fuel=s["fuel"],
            throttle=s["throttle"], gimbal=s["gimbal"],
        ))
        if terminated or truncated:
            outcome = classify_outcome(s, stage)
            break
    else:
        outcome = "TIMEOUT"

    return frames, outcome


# ── Static background dressing ──────────────────────────────────────────────

def add_starfield(ax, bounds, count=260, seed=7):
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1), (z0, z1) = bounds
    xs = rng.uniform(x0, x1, count)
    ys = rng.uniform(y0, y1, count)
    zs = rng.uniform(z0, z1, count)
    sizes = rng.uniform(1.5, 9.0, count) ** 1.3
    alphas = rng.uniform(0.25, 0.9, count)
    colors = [(1, 1, 1, a) for a in alphas]
    ax.scatter(xs, ys, zs, s=sizes, c=colors, linewidths=0, depthshade=False)


def add_planet(ax, pos, radius, color, seed=0):
    """A soft glowing sphere-like blob: a bright core plus a faint halo."""
    x, y, z = pos
    ax.scatter([x], [y], [z], s=radius * 5.5, c=color, alpha=0.18,
               linewidths=0, depthshade=False)
    ax.scatter([x], [y], [z], s=radius * 2.6, c=color, alpha=0.35,
               linewidths=0, depthshade=False)
    ax.scatter([x], [y], [z], s=radius, c=color, alpha=0.95,
               linewidths=0, depthshade=False)


def add_landing_pad(ax, x_offset, y_lane, pad_half, lane_half_depth=3.0):
    xs = [-pad_half, pad_half, pad_half, -pad_half]
    ys = [y_lane - lane_half_depth] * 2 + [y_lane + lane_half_depth] * 2
    zs = [0.0] * 4
    verts = [list(zip(xs, ys, zs))]
    pad = Poly3DCollection(verts, facecolor=PAD_FACE, edgecolor=PAD_COL, linewidth=1.4)
    ax.add_collection3d(pad, autolim=False)

    # Centre marker: a small cross-tick at the precise touchdown target
    tick = pad_half * 0.18
    ax.plot([-tick, tick], [y_lane, y_lane], [0.02, 0.02], color=PAD_COL, linewidth=1.3, alpha=0.9)
    ax.plot([0, 0], [y_lane - tick, y_lane + tick], [0.02, 0.02], color=PAD_COL, linewidth=1.3, alpha=0.9)


# ── 3D rocket model (styled after render.py's pygame rocket) ───────────────
#
# All geometry is built in a local body frame: the long axis runs along
# local z (z=0 at the engine/base, +z toward the nose), and the circular
# cross-section spans local x/y. A body tilt (or gimbal deflection) is a
# rotation about the world depth axis (y), matching render.py's 2D
# `_rotate` helper but carrying the extra (depth) axis through unchanged --
# so the exact same "positive angle tilts the nose toward +x" convention
# used by the physics and the pygame renderer holds here too.

def _place(X, Y, Z, angle_rad, origin):
    """Rotate a local-frame mesh/points about the world depth (y) axis, then translate."""
    ox, oy, oz = origin
    Xr = X * np.cos(angle_rad) + Z * np.sin(angle_rad)
    Zr = -X * np.sin(angle_rad) + Z * np.cos(angle_rad)
    return Xr + ox, Y + oy, Zr + oz


def _cone_mesh(r0, r1, height, n_sides=12, n_levels=3, z0=0.0):
    """Parametric surface for a cylinder (r0==r1) or cone (r0!=r1) along local z."""
    theta = np.linspace(0, 2 * np.pi, n_sides + 1)
    t = np.linspace(0, 1, n_levels)
    Theta, T = np.meshgrid(theta, t)
    R = r0 + (r1 - r0) * T
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = z0 + T * height
    return X, Y, Z


def _fin_quad(phi0, r_body, fin_len, fin_h, z0):
    """One swept trapezoidal fin as 4 local-frame corner points, base at z0."""
    c, s = np.cos(phi0), np.sin(phi0)
    inner_r, outer_r, outer_r_top = r_body, r_body + fin_len, r_body + fin_len * 0.4
    pts = [
        (inner_r * c, inner_r * s, z0),
        (outer_r * c, outer_r * s, z0),
        (outer_r_top * c, outer_r_top * s, z0 + fin_h),
        (inner_r * c, inner_r * s, z0 + fin_h * 1.3),
    ]
    X = np.array([p[0] for p in pts])
    Y = np.array([p[1] for p in pts])
    Z = np.array([p[2] for p in pts])
    return X, Y, Z


def build_rocket_artists(ax, x, y, angle, gimbal, throttle, tick):
    """
    Build one frame's worth of 3D rocket + flame artists at world position
    (x, 0, y). Returns the list of new artists (caller removes the previous
    frame's list before calling again).
    """
    r_body, h_body, h_nose = 1.1, 4.2, 1.8
    artists = []

    def surf(mesh, color, alpha=1.0):
        X, Y, Z = mesh
        s = ax.plot_surface(X, Y, Z, color=color, shade=True, alpha=alpha,
                             linewidth=0, antialiased=True)
        artists.append(s)
        return s

    origin = (x, 0.0, y)

    # Hull: cylinder + shaded nose cone
    body = _place(*_cone_mesh(r_body, r_body, h_body, z0=0.0), angle, origin)
    surf(body, ROCKET_HULL_COL)
    nose = _place(*_cone_mesh(r_body, 0.02, h_nose, z0=h_body), angle, origin)
    surf(nose, ROCKET_SHADE_COL)

    # Engine nozzle stub, flared slightly outward below the base
    nozzle = _place(*_cone_mesh(r_body * 0.5, r_body * 0.62, 0.35, z0=-0.35), angle, origin)
    surf(nozzle, ROCKET_DARK_COL)

    # Fins, four around the base for good coverage from any camera angle
    for phi0 in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        fx, fy, fz = _fin_quad(phi0, r_body, 1.3, 1.1, z0=0.3)
        fx, fy, fz = _place(fx, fy, fz, angle, origin)
        verts = [list(zip(fx, fy, fz))]
        fin = Poly3DCollection(verts, facecolor=ROCKET_HULL_COL,
                                edgecolor=ROCKET_SHADE_COL, linewidth=0.5)
        ax.add_collection3d(fin, autolim=False)
        artists.append(fin)

    # Window: a small flattened disc facing +local-x, bulging just past the
    # hull surface so it doesn't z-fight with the body
    wtheta = np.linspace(0, 2 * np.pi, 14)
    wr = 0.32
    wz0 = h_body * 0.62
    disc_x = np.full_like(wtheta, r_body * 1.02)
    disc_y = wr * np.cos(wtheta)
    disc_z = wz0 + wr * np.sin(wtheta)
    dx, dy, dz = _place(disc_x, disc_y, disc_z, angle, origin)
    verts = [list(zip(dx, dy, dz))]
    window = Poly3DCollection(verts, facecolor=ROCKET_WINDOW_COL,
                               edgecolor=ROCKET_DARK_COL, linewidth=0.6)
    ax.add_collection3d(window, autolim=False)
    artists.append(window)

    # Flame: nested tapered cones, own rotation (nozzle = body tilt + gimbal
    # deflection), with a small time-based flicker for a "living fire" look.
    # Proportioned relative to the body (flame roughly as long as the hull at
    # full throttle, not longer) and brightened so the outer plume doesn't
    # disappear into the black background.
    if throttle > 0.02:
        nozzle_angle = angle + gimbal * MAX_GIMBAL_ANGLE
        flicker = 1.0 + 0.08 * np.sin(tick * 1.7) + 0.04 * np.sin(tick * 4.1)
        flame_len = throttle * 4.6 * flicker
        layers = [
            (1.00, r_body * 0.95, 0.75),
            (0.68, r_body * 0.62, 0.9),
            (0.40, r_body * 0.32, 1.0),
        ]
        for i, (len_scale, base_r, alpha) in enumerate(layers):
            # Negative height: the cone must extend to -z (away from the
            # engine base), not back up into the body.
            mesh = _cone_mesh(base_r, 0.01, -flame_len * len_scale, n_sides=12, n_levels=3, z0=-0.35)
            mesh = _place(*mesh, nozzle_angle, origin)
            surf(mesh, ROCKET_FLAME_COLS[i], alpha=alpha)

        # A bright additive glow near the flame tip for extra "fire" life.
        # Same local -z convention as the flame cones above, so subtract.
        tip_dist = 0.35 + flame_len * 0.75
        tx = origin[0] - np.sin(nozzle_angle) * tip_dist
        tz = origin[2] - np.cos(nozzle_angle) * tip_dist
        sp = ax.scatter([tx], [0.0], [tz], s=220 * throttle, c=[ROCKET_FLAME_COLS[2]],
                         alpha=0.6, linewidths=0, depthshade=False)
        artists.append(sp)

    return artists


def build_single_rocket_animation(frames, outcome, stage, fps, zoom_half=20.0,
                                   azim_start=-50.0, azim_span=40.0, elev=16.0):
    """A zoomed-in, chase-camera animation of a single flight with a full 3D rocket model."""
    cfg = STAGES[stage]
    pad_half = cfg["pad"]
    max_len = len(frames)
    # Make sure the pad's full width always fits the horizontal window,
    # regardless of which stage's (differently-sized) pad is being rendered.
    zoom_half = max(zoom_half, pad_half * 1.2)

    fig = plt.figure(figsize=(11, 9), facecolor=BG_COL)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COL)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(PANE_COL)
        pane.pane.set_edgecolor((0, 0, 0, 0))
        pane._axinfo["grid"]["color"] = GRID_COL
        pane.line.set_color(AXIS_COL)
    ax.tick_params(colors=AXIS_COL, labelsize=8)
    ax.set_xlabel("Horizontal position (m)", color=AXIS_COL, fontsize=9, labelpad=8)
    ax.set_zlabel("Altitude (m)", color=AXIS_COL, fontsize=9, labelpad=6)
    ax.set_yticks([])
    ax.set_box_aspect((1.0, 1.0, 1.0))

    fig.suptitle(f"PPO Rocket Landing — Stage {stage}", color="#e8e8f0",
                 fontsize=15, fontweight="bold", y=0.96)

    # ── Background dressing (drawn once, wide range for camera parallax) ───
    bounds = ((-160, 160), (-150, 400), (0, cfg["alt"] * 1.6))
    add_starfield(ax, bounds, count=220)
    add_planet(ax, (120, 260, cfg["alt"] * 1.3), 900, MOON_COL)
    add_planet(ax, (-130, 120, cfg["alt"] * 1.45), 500, PLANET_COL)
    add_landing_pad(ax, 0.0, 0.0, pad_half, lane_half_depth=pad_half * 0.4)

    # ── Dynamic artists ─────────────────────────────────────────────────────
    cmap = matplotlib.colormaps[THROTTLE_CMAP]
    trail = Line3DCollection([], linewidths=3.0, cmap=cmap, norm=plt.Normalize(0, 1))
    ax.add_collection3d(trail, autolim=False)

    mappable = cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.02, aspect=22)
    cbar.set_label("Throttle", color=AXIS_COL, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=AXIS_COL, labelcolor=AXIS_COL, labelsize=8)

    hud_text = fig.text(0.02, 0.10, "", color="#cfd2de", fontsize=10, family="monospace",
                         va="bottom")
    outcome_label = ax.text(0, 0, 0, "", fontsize=12, fontweight="bold", color=LAND_COL)

    rocket_artists = []

    def update(frame):
        idx = min(frame, max_len - 1)
        segment = frames[:idx + 1]
        cur = segment[-1]

        ax.view_init(elev=elev, azim=azim_start + azim_span * frame / max(max_len - 1, 1))

        # Chase camera: tightly centred on the rocket at altitude, but blends
        # toward a ground-anchored framing (pad visible at the bottom of the
        # window) as it descends, so the pad is clearly in shot well before
        # -- and regardless of whether -- the rocket actually reaches it.
        # The ground-anchored target still keeps headroom above the rocket's
        # *current* altitude (not a fixed height), so the rocket itself never
        # drifts out of frame even if the flight ends well above the ground.
        reveal_start, reveal_full = zoom_half * 13.0, zoom_half * 5.0
        t = float(np.clip((reveal_start - cur["y"]) / (reveal_start - reveal_full), 0.0, 1.0))
        chase_lo, chase_hi = cur["y"] - zoom_half, cur["y"] + zoom_half
        ground_lo = -3.0
        ground_hi = max(ground_lo + zoom_half * 2.2, cur["y"] + zoom_half * 0.6)
        z_lo = max(-3.0, chase_lo * (1 - t) + ground_lo * t)
        z_hi = chase_hi * (1 - t) + ground_hi * t

        ax.set_xlim(cur["x"] - zoom_half, cur["x"] + zoom_half)
        ax.set_zlim(z_lo, z_hi)
        ax.set_ylim(-zoom_half, zoom_half)

        xs = [f["x"] for f in segment]
        ys = [0.0] * len(segment)
        zs = [f["y"] for f in segment]
        throttles = [f["throttle"] for f in segment]
        pts = np.array([xs, ys, zs]).T
        segs = np.stack([pts[:-1], pts[1:]], axis=1) if len(pts) > 1 else np.empty((0, 2, 3))
        trail.set_segments(segs)
        if len(throttles) > 1:
            trail.set_array(np.array(throttles[1:]))

        for a in rocket_artists:
            a.remove()
        rocket_artists.clear()
        rocket_artists.extend(build_rocket_artists(
            ax, cur["x"], cur["y"], cur["angle"], cur["gimbal"], cur["throttle"],
            tick=frame * DT,
        ))

        hud_text.set_text(
            f"t = {cur['t']:6.2f} s\n"
            f"alt      {cur['y']:7.1f} m\n"
            f"vx       {cur['vx']:+6.1f} m/s\n"
            f"vy       {cur['vy']:+6.1f} m/s\n"
            f"throttle {cur['throttle']*100:5.1f} %\n"
            f"fuel     {cur['fuel']:6.1f} kg"
        )

        if idx == max_len - 1:
            col = outcome_color(outcome)
            outcome_label.set_text(f"  {outcome}")
            outcome_label.set_color(col)
            outcome_label.set_position((cur["x"], 0.0))
            outcome_label.set_3d_properties(cur["y"] + zoom_half * 0.25, zdir="z")

        return []

    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=1000 / fps, blit=False)
    return fig, ani


# ── Animation ────────────────────────────────────────────────────────────────

def build_animation(episodes, stage, fps, azim_start, azim_span, elev):
    cfg = STAGES[stage]
    pad_half = cfg["pad"]
    n_ep = len(episodes)

    max_abs_x = max(
        (abs(f["x"]) for frames, _ in episodes for f in frames), default=pad_half
    )
    max_alt = max(
        (f["y"] for frames, _ in episodes for f in frames), default=cfg["alt"]
    )
    max_abs_x = max(max_abs_x, pad_half) * 1.15
    lane_spacing = max(45.0, max_abs_x * 1.3)
    lane_offsets = [i * lane_spacing for i in range(n_ep)]

    max_len = max(len(frames) for frames, _ in episodes)

    fig = plt.figure(figsize=(13, 8.5), facecolor=BG_COL)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(BG_COL)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(PANE_COL)
        pane.pane.set_edgecolor((0, 0, 0, 0))
        pane._axinfo["grid"]["color"] = GRID_COL
        pane.line.set_color(AXIS_COL)
    ax.tick_params(colors=AXIS_COL, labelsize=8)
    ax.set_xlabel("Horizontal position (m)", color=AXIS_COL, fontsize=9, labelpad=10)
    ax.set_ylabel("Episode", color=AXIS_COL, fontsize=9, labelpad=10)
    ax.set_zlabel("Altitude (m)", color=AXIS_COL, fontsize=9, labelpad=6)

    ax.set_xlim(-max_abs_x, max_abs_x)
    ax.set_ylim(-lane_spacing * 0.5, lane_offsets[-1] + lane_spacing * 0.5)
    ax.set_zlim(0, max_alt * 1.08)
    ax.set_yticks(lane_offsets)
    ax.set_yticklabels([f"#{i+1}" for i in range(n_ep)])
    ax.set_box_aspect((2.0, 1.0, 1.1))

    fig.suptitle(f"PPO Rocket Landing — Stage {stage}", color="#e8e8f0",
                 fontsize=15, fontweight="bold", y=0.96)
    fig.text(0.5, 0.915, f"{n_ep} evaluation episode(s)  ·  deterministic policy",
              color=AXIS_COL, fontsize=9, ha="center")

    # ── Background dressing (drawn once) ────────────────────────────────────
    bounds = (
        (-max_abs_x * 1.4, max_abs_x * 1.4),
        (-lane_spacing, lane_offsets[-1] + lane_spacing),
        (0, max_alt * 1.6),
    )
    add_starfield(ax, bounds)
    add_planet(ax, (max_abs_x * 1.05, lane_offsets[-1] * 0.7, max_alt * 1.35), 900, MOON_COL)
    add_planet(ax, (-max_abs_x * 1.1, lane_offsets[-1] * 0.25, max_alt * 1.5), 500, PLANET_COL)

    for lane in lane_offsets:
        add_landing_pad(ax, 0.0, lane, pad_half)

    # ── Dynamic artists (updated per frame) ─────────────────────────────────
    cmap = matplotlib.colormaps[THROTTLE_CMAP]
    trail_collections, rocket_dots, thrust_lines, outcome_labels = [], [], [], []
    for i in range(n_ep):
        lc = Line3DCollection([], linewidths=2.4, cmap=cmap, norm=plt.Normalize(0, 1))
        ax.add_collection3d(lc, autolim=False)
        trail_collections.append(lc)

        (dot,) = ax.plot([], [], [], marker="o", markersize=6.5,
                          color=ROCKET_COL, markeredgecolor="#9a9aa5", linestyle="")
        rocket_dots.append(dot)

        (tline,) = ax.plot([], [], [], color=THRUST_COL, linewidth=2.2, alpha=0.9)
        thrust_lines.append(tline)

        label = ax.text(0, lane_offsets[i], max_alt * 0.05, "", fontsize=10,
                         fontweight="bold", color=LAND_COL)
        outcome_labels.append(label)

    mappable = cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.02, aspect=22)
    cbar.set_label("Throttle", color=AXIS_COL, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=AXIS_COL, labelcolor=AXIS_COL, labelsize=8)

    clock_text = fig.text(0.02, 0.04, "", color="#cfd2de", fontsize=10, family="monospace")
    finished = [False] * n_ep

    def flame_tip(f):
        nozzle_angle = f["angle"] + f["gimbal"] * MAX_GIMBAL_ANGLE
        length = 4.0 + 14.0 * f["throttle"]
        return -np.sin(nozzle_angle) * length, -np.cos(nozzle_angle) * length

    def update(frame):
        ax.view_init(elev=elev, azim=azim_start + azim_span * frame / max(max_len - 1, 1))

        t_now = frame * DT
        for i, (frames, outcome) in enumerate(episodes):
            idx = min(frame, len(frames) - 1)
            segment = frames[:idx + 1]
            xs = [f["x"] for f in segment]
            ys = [lane_offsets[i]] * len(segment)
            zs = [f["y"] for f in segment]
            throttles = [f["throttle"] for f in segment]

            pts = np.array([xs, ys, zs]).T
            segs = np.stack([pts[:-1], pts[1:]], axis=1) if len(pts) > 1 else np.empty((0, 2, 3))
            trail_collections[i].set_segments(segs)
            if len(throttles) > 1:
                trail_collections[i].set_array(np.array(throttles[1:]))

            cur = segment[-1]
            rocket_dots[i].set_data_3d([cur["x"]], [lane_offsets[i]], [cur["y"]])

            dx, dz = flame_tip(cur)
            thrust_lines[i].set_data_3d(
                [cur["x"], cur["x"] + dx], [lane_offsets[i]] * 2, [cur["y"], cur["y"] + dz],
            )
            thrust_lines[i].set_alpha(0.25 + 0.65 * cur["throttle"])

            if idx == len(frames) - 1 and not finished[i]:
                finished[i] = True
                col = outcome_color(outcome)
                outcome_labels[i].set_text(f"  #{i+1}: {outcome}")
                outcome_labels[i].set_color(col)
                outcome_labels[i].set_position((cur["x"], lane_offsets[i]))
                outcome_labels[i].set_3d_properties(cur["y"] + max_alt * 0.05, zdir="z")
            if finished[i]:
                rocket_dots[i].set_color(outcome_color(outcome))

        clock_text.set_text(f"t = {t_now:5.2f} s")
        return []

    ani = animation.FuncAnimation(fig, update, frames=max_len, interval=1000 / fps, blit=False)
    return fig, ani


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=None,
                         help="curriculum stage to evaluate (default: read from current_stage.txt)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="base seed; episode i uses seed+i")
    parser.add_argument("--seeds", type=str, default=None,
                         help="comma-separated explicit seed list, overrides --seed/--episodes")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--norm", type=str, default=DEFAULT_NORM_PATH)
    parser.add_argument("--max-steps", type=int, default=500,
                         help="safety cap on logged steps per episode (sim seconds = max_steps*0.05)")
    parser.add_argument("--out", type=str, default="rocket/trajectory_3d.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--elev", type=float, default=None,
                         help="camera elevation (default 20 for overview, 16 for single-rocket close-up)")
    parser.add_argument("--azim-start", type=float, default=None,
                         help="starting azimuth (default -60 overview, -50 close-up)")
    parser.add_argument("--azim-span", type=float, default=None,
                         help="total azimuth rotation in degrees (default 70 overview, 40 close-up)")
    parser.add_argument("--zoom", type=float, default=16.0,
                         help="single-rocket mode only: camera half-window size in metres")
    args = parser.parse_args()

    stage = args.stage if args.stage is not None else read_current_stage()
    print(f"Loading PPO agent (stage {stage}) from {args.model}...")
    model, vec_normalize = load_agent(stage, args.model, args.norm)

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else \
            [args.seed + i for i in range(args.episodes)]

    episodes = []
    for i, seed in enumerate(seeds):
        print(f"Running evaluation episode {i+1}/{len(seeds)} (seed={seed})...")
        frames, outcome = collect_episode(model, vec_normalize, stage, seed, args.max_steps)
        print(f"  -> {len(frames)} steps, outcome: {outcome}")
        episodes.append((frames, outcome))

    print("Building 3D animation...")
    if len(episodes) == 1:
        frames, outcome = episodes[0]
        fig, ani = build_single_rocket_animation(
            frames, outcome, stage, args.fps, zoom_half=args.zoom,
            azim_start=args.azim_start if args.azim_start is not None else -50.0,
            azim_span=args.azim_span if args.azim_span is not None else 40.0,
            elev=args.elev if args.elev is not None else 16.0,
        )
    else:
        fig, ani = build_animation(
            episodes, stage, args.fps,
            azim_start=args.azim_start if args.azim_start is not None else -60.0,
            azim_span=args.azim_span if args.azim_span is not None else 70.0,
            elev=args.elev if args.elev is not None else 20.0,
        )

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if not ffmpeg_path:
        print("ERROR: no ffmpeg found (system PATH or `pip install imageio-ffmpeg`). "
              "Cannot export MP4.")
        sys.exit(1)
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Encoding {out_path} ({args.fps} fps, {args.dpi} dpi)...")
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=4000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    ani.save(out_path, writer=writer, dpi=args.dpi, savefig_kwargs={"facecolor": BG_COL})
    plt.close(fig)
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
