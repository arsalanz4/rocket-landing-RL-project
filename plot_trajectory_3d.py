"""
Cinematic 3D trajectory renderer for the Rocket Landing environment.

Simulates one episode (trained PPO policy if available, otherwise the
hand-coded heuristic pilot) and renders it as a matplotlib 3D animation:
  - a low-poly shaded rocket body with nose cone, fins and engine nozzle
  - a layered, flickering engine flame with an additive glow bloom
  - a heat-graded contrail that fades and cools behind the rocket
  - a fixed full-height altitude axis (the whole flight is always visible)
    PLUS a "landing cam" inset locked on the pad so touchdown is always
    readable up close, even though the main view never re-scales.

Usage
-----
    python plot_trajectory_3d.py --stage 7 \
        --model best_model --vec-normalize vec_normalize.pkl \
        --output trajectory_3d.mp4

    python plot_trajectory_3d.py --stage 7 --preview   # a few PNG stills, fast

Omitting --model falls back to the hand-coded heuristic pilot, which needs
no ML dependencies (numpy + gymnasium only) so the visuals can be iterated
on without stable-baselines3 / torch installed.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rocket_env import (
    RocketLandingEnv, STAGES,
    MAX_LANDING_VY, MAX_LANDING_VX,
    MAX_GIMBAL_ANGLE, MAX_TILT,
)

DT = 0.05
SIM_FPS = 20

# ── Palette ─────────────────────────────────────────────────────────────────
BG_COL       = "#03040c"
PAD_COL      = (0.86, 0.70, 0.16)
PAD_GLOW_COL = (1.00, 0.80, 0.25)
BODY_LIT     = np.array([0.86, 0.87, 0.90])
STRIPE_COL   = np.array([0.75, 0.12, 0.10])
FIN_COL      = np.array([0.55, 0.56, 0.62])
NOZZLE_COL   = np.array([0.18, 0.18, 0.21])
WINDOW_COL   = np.array([0.45, 0.80, 0.95])
FLAME_CORE   = np.array([1.00, 0.98, 0.85])
FLAME_MID    = np.array([1.00, 0.66, 0.12])
FLAME_OUTER  = np.array([0.85, 0.20, 0.05])
CONTRAIL_HOT = np.array([1.00, 0.75, 0.35])
CONTRAIL_COLD = np.array([0.55, 0.58, 0.68])
TEXT_COL     = "#dcdfe8"
TITLE_COL    = "#b9c4ff"
LIGHT_DIR    = np.array([0.45, -0.55, 0.71])
LIGHT_DIR    = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


# ── Episode simulation ───────────────────────────────────────────────────────

def heuristic_action(obs):
    alt, vy, angle, ang_vel = obs[1], obs[3], obs[4], obs[5]
    throttle = 1.0 if (vy < -15 or alt < 100) else 0.3
    gimbal = float(np.clip(angle * 3.0 + ang_vel * 1.0, -1.0, 1.0))
    return np.array([throttle, gimbal], dtype=np.float32)


def load_ppo_policy(model_path, vec_normalize_path, stage):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    vec_env = make_vec_env(lambda: RocketLandingEnv(stage=stage), n_envs=1)
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training, vec_env.norm_reward = False, False
    model = PPO.load(model_path, env=vec_env)

    def policy(_env, obs):
        obs_n = vec_env.normalize_obs(obs.reshape(1, -1))[0]
        action, _ = model.predict(obs_n, deterministic=True)
        return action

    return policy


def simulate_episode(stage, policy, max_steps=None, seed=None):
    """Run one episode and return a list of state-dict snapshots (deep copies)."""
    env = RocketLandingEnv(stage=stage)
    obs, _ = env.reset(seed=seed)
    history = []
    done = False
    steps = 0
    cap = max_steps or 2000
    while not done and steps < cap:
        action = policy(env, obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        history.append(dict(env._state))
        steps += 1

    s = env._state
    speed_ok = abs(s["vy"]) <= MAX_LANDING_VY and abs(s["vx"]) <= MAX_LANDING_VX
    upright = abs(s["angle"]) <= MAX_TILT
    pad_half = STAGES[env.stage]["pad"]
    on_pad = abs(s["x"]) <= pad_half
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
    return history, outcome, pad_half


# ── Low-poly rocket geometry ──────────────────────────────────────────────────

N_SIDES = 10


def _ring(radius, z, cx=0.0, cy=0.0, n=N_SIDES):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th),
                      np.full(n, z)], axis=1)


def _shade(base_rgb, normal):
    n = normal / (np.linalg.norm(normal) + 1e-9)
    diffuse = max(0.0, float(np.dot(n, LIGHT_DIR)))
    lum = 0.35 + 0.75 * diffuse
    return tuple(np.clip(base_rgb * lum, 0, 1))


def _side_faces(ring_lo, ring_hi, base_col):
    """Quad strip between two rings, each face shaded by its outward normal."""
    n = len(ring_lo)
    faces, colors = [], []
    for i in range(n):
        j = (i + 1) % n
        quad = np.array([ring_lo[i], ring_lo[j], ring_hi[j], ring_hi[i]])
        centre = quad.mean(axis=0)
        normal = np.array([centre[0], centre[1], 0.0])
        faces.append(quad)
        colors.append(_shade(base_col, normal))
    return faces, colors


def _cone_faces(ring, apex, base_col):
    n = len(ring)
    faces, colors = [], []
    for i in range(n):
        j = (i + 1) % n
        tri = np.array([ring[i], ring[j], apex])
        centre = tri.mean(axis=0)
        normal = np.array([centre[0] - apex[0] * 0, centre[1], 0.35])
        faces.append(tri)
        colors.append(_shade(base_col, normal))
    return faces, colors


def build_rocket_mesh(length=8.6, radius=1.1):
    """Body built pointing +z (nose up), base (engine) at z=0. Returns face lists."""
    nose_h  = length * 0.28
    body_h  = length * 0.72
    ring_base = _ring(radius, 0.0)
    ring_top  = _ring(radius, body_h)
    apex      = np.array([0.0, 0.0, body_h + nose_h])

    body_faces, body_colors = _side_faces(ring_base, ring_top, BODY_LIT)
    # Trailing-edge shadow stripe: darken the faces facing away from light
    for i, c in enumerate(body_colors):
        if c[0] + c[1] + c[2] < 0.9:
            body_colors[i] = tuple(np.clip(np.array(c) * 0.8, 0, 1))

    nose_faces, nose_colors = _cone_faces(ring_top, apex, BODY_LIT)

    # Accent stripe: one ring band near the top of the body
    stripe_lo = _ring(radius * 1.001, body_h * 0.72)
    stripe_hi = _ring(radius * 1.001, body_h * 0.86)
    stripe_faces, stripe_colors = _side_faces(stripe_lo, stripe_hi, STRIPE_COL)

    # Window: small cyan glass patch roughly halfway up the body, facing +x
    win_z = body_h * 0.55
    win_faces, win_colors = [], []
    wx = radius * 1.0
    hw = radius * 0.28
    quad = np.array([
        [wx, -hw, win_z - hw], [wx, hw, win_z - hw],
        [wx, hw, win_z + hw], [wx, -hw, win_z + hw],
    ])
    win_faces.append(quad)
    win_colors.append(tuple(WINDOW_COL))

    # Three fins at 0/120/240 degrees around the base
    fin_faces, fin_colors = [], []
    fin_span, fin_h, fin_sweep = radius * 1.9, length * 0.22, length * 0.16
    for ang in (0, 2 * np.pi / 3, 4 * np.pi / 3):
        ca, sa = np.cos(ang), np.sin(ang)
        root_lo = np.array([radius * ca, radius * sa, 0.0])
        root_hi = np.array([radius * ca, radius * sa, fin_h])
        tip     = np.array([fin_span * ca, fin_span * sa, fin_h * 0.15])
        tip_back = np.array([(radius + fin_sweep * 0.3) * ca,
                              (radius + fin_sweep * 0.3) * sa, -fin_sweep])
        tri = np.array([root_lo, root_hi, tip, tip_back])
        centre = tri.mean(axis=0)
        normal = np.array([ca, sa, 0.0])
        fin_faces.append(tri)
        fin_colors.append(_shade(FIN_COL, normal))

    # Nozzle: small inverted frustum below the base
    nozzle_top = _ring(radius * 0.55, 0.0)
    nozzle_bot = _ring(radius * 0.80, -fin_h * 0.5)
    nozzle_faces, nozzle_colors = _side_faces(nozzle_top, nozzle_bot, NOZZLE_COL)

    return dict(
        body=(body_faces, body_colors),
        nose=(nose_faces, nose_colors),
        stripe=(stripe_faces, stripe_colors),
        window=(win_faces, win_colors),
        fins=(fin_faces, fin_colors),
        nozzle=(nozzle_faces, nozzle_colors),
    )


def transform_faces(faces, angle_rad, origin):
    """Rotate about the depth (y) axis by angle_rad (body tilt) and translate."""
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    out = []
    for f in faces:
        out.append((f @ R.T) + origin)
    return out


# ── Flame + glow ──────────────────────────────────────────────────────────────

def build_flame(origin, angle_rad, gimbal, throttle, frame_idx, radius=1.1):
    """Layered cone flame + a soft point-cloud glow bloom, both already
    rotated/translated into world space."""
    if throttle < 0.02:
        return [], [], np.empty((0, 3)), np.empty((0,))

    nozzle_angle = angle_rad + gimbal * MAX_GIMBAL_ANGLE
    flicker = 1.0 + 0.08 * np.sin(frame_idx * 0.9) + 0.04 * np.sin(frame_idx * 2.3 + 1.0)
    length = radius * 6.36 * throttle * flicker
    dirv = np.array([np.sin(nozzle_angle), 0.0, -np.cos(nozzle_angle)])

    layers = [
        (length * 1.00, radius * 0.60, FLAME_OUTER),
        (length * 0.68, radius * 0.38, FLAME_MID),
        (length * 0.38, radius * 0.20, FLAME_CORE),
    ]
    faces, colors = [], []
    for layer_len, layer_r, col in layers:
        ring = _ring(layer_r, 0.0) @ np.eye(3)  # local ring in xy at z=0
        # orient ring around the flame axis: build local basis
        up = dirv
        arb = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(up, arb); e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        ring_world = origin + ring[:, 0:1] * e1 + ring[:, 1:2] * e2
        tip = origin + up * layer_len
        n = len(ring_world)
        for i in range(n):
            j = (i + 1) % n
            tri = np.array([ring_world[i], ring_world[j], tip])
            faces.append(tri)
            colors.append(tuple(col))

    # Additive-looking glow: scattered translucent points near the nozzle/plume
    rng = np.random.default_rng(frame_idx)
    n_pts = 26
    t = rng.uniform(0.0, 1.0, n_pts) ** 0.6
    spread = radius * 0.5 * (1 - t) + radius * 0.15
    ang = rng.uniform(0, 2 * np.pi, n_pts)
    up = dirv
    arb = np.array([0.0, 1.0, 0.0]) if abs(up[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(up, arb); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    pts = (origin[None, :] + (up[None, :] * (t * length)[:, None])
           + e1[None, :] * (spread * np.cos(ang))[:, None]
           + e2[None, :] * (spread * np.sin(ang))[:, None])
    glow_alpha = (1 - t) * 0.55 * min(1.0, throttle * 1.3)
    return faces, colors, pts, glow_alpha


# ── Contrail ──────────────────────────────────────────────────────────────────

def build_contrail(positions, throttles, max_len=140):
    pts = positions[-max_len:]
    thr = throttles[-max_len:]
    if len(pts) < 2:
        return [], np.empty((0, 4)), np.empty((0,))
    n = len(pts)
    segs = [[pts[i], pts[i + 1]] for i in range(n - 1)]
    colors = np.zeros((n - 1, 4))
    widths = np.zeros(n - 1)
    for i in range(n - 1):
        s = i / max(n - 2, 1)  # 0 = old/tail, 1 = new/near-rocket
        heat = 0.25 + 0.75 * thr[i + 1]
        rgb = CONTRAIL_COLD * (1 - heat) + CONTRAIL_HOT * heat
        colors[i, :3] = rgb
        colors[i, 3] = 0.05 + 0.55 * s
        widths[i] = 0.6 + 2.2 * s
    return segs, colors, widths


# ── Scene setup ───────────────────────────────────────────────────────────────

def style_axes(ax, xlim, ylim, zlim, elev, azim):
    ax.set_facecolor(BG_COL)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.view_init(elev=elev, azim=azim)
    ax.xaxis.pane.set_facecolor((0.04, 0.045, 0.09, 1.0))
    ax.yaxis.pane.set_facecolor((0.04, 0.045, 0.09, 1.0))
    ax.zaxis.pane.set_facecolor((0.03, 0.035, 0.07, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor((0.25, 0.27, 0.4, 0.6))
        axis.line.set_color((0.4, 0.42, 0.55))
        axis._axinfo["grid"]["color"] = (0.25, 0.27, 0.4, 0.35)
    ax.tick_params(colors=(0.55, 0.57, 0.68), labelsize=8)


def draw_pad(ax, pad_half, glow=0.0, n=24):
    xs = np.linspace(-pad_half, pad_half, n)
    ys = np.linspace(-pad_half, pad_half, n)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, color=PAD_COL, alpha=0.32 + 0.25 * glow,
                     linewidth=0, shade=False, zorder=1)
    # Border
    bx = [-pad_half, pad_half, pad_half, -pad_half, -pad_half]
    by = [-pad_half, -pad_half, pad_half, pad_half, -pad_half]
    ax.plot(bx, by, [0] * 5, color=PAD_COL, linewidth=2.0, alpha=0.9)
    # Bullseye rings
    for r in (pad_half * 0.5, pad_half * 0.28):
        th = np.linspace(0, 2 * np.pi, 40)
        ax.plot(r * np.cos(th), r * np.sin(th), np.zeros(40),
                color=PAD_GLOW_COL, linewidth=1.2, alpha=0.65 + 0.3 * glow)
    if glow > 0.02:
        th = np.linspace(0, 2 * np.pi, 30)
        for r in np.linspace(pad_half * 0.1, pad_half * 0.9, 5):
            ax.plot(r * np.cos(th), r * np.sin(th), np.zeros(30),
                    color=PAD_GLOW_COL, linewidth=6, alpha=0.05 * glow)


def draw_mesh(ax, mesh, angle, origin, scale=1.0, parts=None):
    for name, (faces, colors) in mesh.items():
        if parts is not None and name not in parts:
            continue
        scaled = [f * scale for f in faces]
        wf = transform_faces(scaled, angle, origin)
        pc = Poly3DCollection(wf, facecolors=colors, edgecolors=(0, 0, 0, 0.25),
                               linewidths=0.25, zorder=5)
        ax.add_collection3d(pc)


def _populate_scene(ax, mesh, pad_half, origin, angle, gimbal, throttle, idx,
                     positions_slice, throttles_slice, scale, glow, pt_size):
    segs, colors, widths = build_contrail(positions_slice, throttles_slice)
    draw_pad(ax, pad_half, glow=glow)
    if segs:
        ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=widths, zorder=4))
    draw_mesh(ax, mesh, angle, origin, scale=scale)
    ffaces, fcolors, gpts, galpha = build_flame(
        origin, angle, gimbal, throttle, idx, radius=1.1 * scale)
    if ffaces:
        pcf = Poly3DCollection(ffaces, facecolors=fcolors, edgecolors=None, zorder=6)
        pcf.set_alpha(0.85)
        ax.add_collection3d(pcf)
        if len(gpts):
            ax.scatter(gpts[:, 0], gpts[:, 1], gpts[:, 2],
                       s=pt_size, c=[FLAME_MID],
                       alpha=float(np.mean(galpha)) if len(galpha) else 0,
                       linewidths=0, depthshade=False, zorder=7)


def populate_rocket_cam_scene(ax, mesh, pad_half, origin, angle, gimbal, throttle, idx, scale):
    """Onboard-camera look: mounted low on the booster looking straight down
    past the engine skirt at the pad rushing up, SpaceX-landing-video style.
    Only the nozzle is drawn (the camera is behind/below the rest of the
    body) plus the flame firing away from the viewer toward the ground."""
    draw_pad(ax, pad_half, glow=0.0)
    draw_mesh(ax, mesh, angle, origin, scale=scale, parts={"nozzle"})
    ffaces, fcolors, gpts, galpha = build_flame(
        origin, angle, gimbal, throttle, idx, radius=1.1 * scale)
    if ffaces:
        pcf = Poly3DCollection(ffaces, facecolors=fcolors, edgecolors=None, zorder=6)
        pcf.set_alpha(0.85)
        ax.add_collection3d(pcf)
        if len(gpts):
            ax.scatter(gpts[:, 0], gpts[:, 1], gpts[:, 2],
                       s=50, c=[FLAME_MID],
                       alpha=float(np.mean(galpha)) if len(galpha) else 0,
                       linewidths=0, depthshade=False, zorder=7)


def render_episode(history, outcome, pad_half, stage, out_path, preview=False,
                    preview_frames=(0.25, 0.55, 0.85, 0.98), fig_w=11.0, fig_h=9.0, dpi=150):
    mesh = build_rocket_mesh()
    n = len(history)
    alt_max = max(s["y"] for s in history) * 1.08
    alt_max = max(alt_max, 40.0)
    x_excursion = max(1.0, max(abs(s["x"]) for s in history))
    x_half = max(pad_half * 1.6, x_excursion * 1.25)

    positions = [(s["x"], 0.0, s["y"]) for s in history]
    throttles = [s["throttle"] for s in history]

    # The rocket is physically ~9m tall; rendered at true scale it is invisible
    # against a multi-hundred-metre-tall flight envelope, so the wide shot uses
    # an exaggerated visual scale while the tight landing-cam inset stays close
    # to true proportions.
    base_len = 8.6
    main_scale = float(np.clip((alt_max * 0.10) / base_len, 4.5, 10.0))
    inset_scale = float(np.clip((60 * 0.30) / base_len, 1.4, 2.8))

    # Rendered as two independent Figures and composited pixel-wise: matplotlib's
    # mplot3d has a bug where a second Axes3D sharing a Figure corrupts the first
    # Axes3D's z-order/clipping (collections silently vanish past a few redraws).
    w_px, h_px = int(fig_w * dpi), int(fig_h * dpi)
    inset_w_in, inset_h_in = 3.6, 2.8
    iw_px, ih_px = int(inset_w_in * dpi), int(inset_h_in * dpi)
    inset_margin = 18

    writer = None
    if not preview:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, SIM_FPS, (w_px, h_px))

    frame_indices = range(n)
    if preview:
        frame_indices = sorted(set(int(f * (n - 1)) for f in preview_frames))

    for idx in frame_indices:
        s = history[idx]
        origin = np.array([s["x"], 0.0, s["y"]])
        angle = s["angle"]
        proximity = max(0.0, 1 - s["y"] / 30.0)
        glow = min(1.0, proximity) * min(1.0, s["throttle"] * 1.4 + 0.15 * proximity)
        pos_slice = positions[:idx + 1]
        thr_slice = throttles[:idx + 1]

        # ── Main wide shot ──────────────────────────────────────────────────
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=BG_COL)
        ax = fig.add_axes([0.03, 0.05, 0.94, 0.90], projection="3d")
        style_axes(ax, (-x_half, x_half), (-x_half * 0.35, x_half * 0.35), (0, alt_max),
                   14, -60 + 10 * np.sin(idx * 0.01))
        _populate_scene(ax, mesh, pad_half, origin, angle, s["gimbal"], s["throttle"], idx,
                        pos_slice, thr_slice, main_scale, glow, pt_size=26)
        ax.set_title("PPO Rocket Landing — Stage %d" % stage, color=TITLE_COL,
                      fontsize=15, fontweight="bold", pad=0)
        ax.set_xlabel("Horizontal position (m)", color=TEXT_COL, fontsize=9)
        ax.set_zlabel("Altitude (m)", color=TEXT_COL, fontsize=9)

        t = idx * DT
        hud = (f"t = {t:6.2f} s\n"
               f"alt   {s['y']:8.1f} m\n"
               f"vx    {s['vx']:+8.1f} m/s\n"
               f"vy    {s['vy']:+8.1f} m/s\n"
               f"throttle {s['throttle']*100:5.1f} %\n"
               f"fuel  {s['fuel']:8.1f} kg")
        fig.text(0.015, 0.95, hud, color=TEXT_COL, family="monospace",
                  fontsize=10, va="top")
        if idx == n - 1:
            col = "#4fe07a" if "LAND" in outcome else "#ff5a3a"
            fig.text(0.5, 0.5, outcome, color=col, fontsize=34, fontweight="bold",
                      ha="center", va="center")

        fig.canvas.draw()
        main_buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)

        # ── Rocket-cam inset: onboard camera looking straight down past the
        # engine at the pad, the ground filling more of frame as it descends
        # (SpaceX booster landing-video style) ──────────────────────────────
        fig_i = plt.figure(figsize=(inset_w_in, inset_h_in), dpi=dpi, facecolor=BG_COL)
        ax_i = fig_i.add_axes([0.05, 0.02, 0.92, 0.90], projection="3d")
        cam_span = float(np.clip(s["y"] * 0.55 + 8, 10, 200))
        style_axes(ax_i, (s["x"] - cam_span, s["x"] + cam_span),
                   (-cam_span, cam_span), (0, s["y"] + cam_span * 0.2), 82, -50)
        populate_rocket_cam_scene(ax_i, mesh, pad_half, origin, angle, s["gimbal"],
                                   s["throttle"], idx, inset_scale)
        ax_i.set_title("Rocket cam", color=(0.72, 0.76, 0.92), fontsize=10)
        fig_i.canvas.draw()
        inset_buf = np.asarray(fig_i.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig_i)

        # Composite: paste the inset into the bottom-left corner with a border
        # (HUD text lives top-left, so this keeps both clear of the main plot's
        # axis tick labels along the bottom and right edges).
        x0 = inset_margin
        y0 = h_px - ih_px - inset_margin
        main_buf[y0 - 2:y0 + ih_px + 2, x0 - 2:x0 + iw_px + 2] = (90, 96, 130)
        main_buf[y0:y0 + ih_px, x0:x0 + iw_px] = inset_buf

        if preview:
            from PIL import Image
            png_path = out_path.replace(".mp4", f"_f{idx}.png")
            Image.fromarray(main_buf).save(png_path)
            print("wrote", png_path)
        else:
            bgr = main_buf[:, :, ::-1]
            writer.write(np.ascontiguousarray(bgr))
            if idx % 50 == 0:
                print(f"  frame {idx}/{n}")

    if writer is not None:
        writer.release()
        print(f"Saved {out_path}  ({n} frames, {n / SIM_FPS:.1f}s @ {SIM_FPS}fps)")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=7)
    ap.add_argument("--model", type=str, default=None,
                     help="path to best_model WITHOUT .zip; omit for heuristic pilot")
    ap.add_argument("--vec-normalize", type=str, default=None)
    ap.add_argument("--output", type=str, default="trajectory_3d.mp4")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--preview", action="store_true",
                     help="dump a handful of PNG stills instead of a full video (fast)")
    args = ap.parse_args()

    if args.model:
        if not args.vec_normalize:
            ap.error("--vec-normalize is required with --model")
        policy = load_ppo_policy(args.model, args.vec_normalize, args.stage)
    else:
        def policy(_env, obs):
            return heuristic_action(obs)

    print(f"Simulating stage {args.stage} episode...")
    history, outcome, pad_half = simulate_episode(
        args.stage, policy, max_steps=args.max_steps, seed=args.seed)
    print(f"  {len(history)} steps -> {outcome}")

    render_episode(history, outcome, pad_half, args.stage, args.output,
                   preview=args.preview)


if __name__ == "__main__":
    main()
