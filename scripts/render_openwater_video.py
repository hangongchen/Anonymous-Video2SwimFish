#!/usr/bin/env python
"""Render an OPEN-WATER swim video from a recorded rollout (play.py --record_demo npz).

Two panels, because they answer different questions:
  LEFT  : fixed world 3D view framed on the whole trajectory + trail  -> does it TRAVEL?
  RIGHT : top-down, re-centred + heading-aligned on the fish          -> does it UNDULATE like a fish?
No tank wireframe (open water), fish drawn large, speed/time readout.
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def rot_mat(q):
    """q (4,) wxyz -> 3x3 body->world rotation."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_apply_inv(q, v):
    """q (4,) wxyz; v (N,3) world -> body frame (v @ R == R.T @ v per row)."""
    return v @ rot_mat(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--title", default="open-water swim")
    args = ap.parse_args()

    d = np.load(args.demo)
    pc = np.asarray(d["pc"], np.float64)          # (T,N,3) world FEM nodes
    root = np.asarray(d["root"], np.float64)      # (T,13)
    T = len(pc)
    pos = root[:, 0:3]
    dt = float(d["step_dt"]) if "step_dt" in d.files else 1.0 / args.fps
    speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt]
    # light smoothing: per-frame FEM jitter otherwise makes the readout unreadable
    k = 9
    speed = np.convolve(speed, np.ones(k) / k, mode="same")

    # world-view bounds from the whole trajectory + body, with margin
    allp = pc.reshape(-1, 3)
    lo, hi = allp.min(0), allp.max(0)
    ctr = (lo + hi) / 2
    span = float(np.max(hi - lo)) * 0.6 + 0.15

    # body-frame extent for the follow panel (fixed zoom across the clip)
    body_all = np.stack([quat_apply_inv(root[t, 3:7], pc[t] - pos[t]) for t in range(T)])
    bmax = float(np.abs(body_all[:, :, :2]).max()) * 1.15

    # heading (body -X), pitch, and the terminal hydro-lift "launch" phase.
    # A nose-up body at high angle of attack gets a huge analytic-lift force; that is a
    # force-model artifact, not swimming, so flag it rather than letting it read as speed.
    fwd = np.stack([rot_mat(root[t, 3:7]) @ np.array([-1.0, 0.0, 0.0]) for t in range(T)])
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 2], -1, 1)))
    steady = np.median(speed[int(0.1 * T):int(0.8 * T)])
    launch = speed > max(3.0 * steady, 0.6)

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=None)
    fig = plt.figure(figsize=(12.4, 5.6))
    for t in range(T):
        # ---- LEFT: fixed world 3D + trail ----
        axL = fig.add_subplot(1, 2, 1, projection="3d")
        axL.plot(pos[:t + 1, 0], pos[:t + 1, 1], pos[:t + 1, 2], color="#2e86de", lw=1.6, alpha=0.85)
        axL.scatter(pc[t, :, 0], pc[t, :, 1], pc[t, :, 2], c=body_all[t][:, 0], cmap="autumn", s=16)
        axL.set_xlim(ctr[0] - span, ctr[0] + span)
        axL.set_ylim(ctr[1] - span, ctr[1] + span)
        axL.set_zlim(ctr[2] - span * 0.6, ctr[2] + span * 0.6)
        axL.set_box_aspect((1, 1, 0.6))
        axL.view_init(elev=24, azim=-58)       # FIXED camera: translation is visible
        axL.set_xticks([]); axL.set_yticks([]); axL.set_zticks([])
        # kill the 3D panes: open water has NO tank, and a drawn box reads as one
        for pane in (axL.xaxis, axL.yaxis, axL.zaxis):
            pane.pane.set_visible(False)
            pane._axinfo["grid"]["linewidth"] = 0.0
        axL.set_title(f"world view + trail   path={np.linalg.norm(np.diff(pos[:t+1],axis=0),axis=1).sum():.2f} m",
                      fontsize=11)

        # ---- RIGHT: heading-aligned top-down (undulation) ----
        axR = fig.add_subplot(1, 2, 2)
        b = body_all[t]
        axR.scatter(b[:, 0], b[:, 1], c=b[:, 0], cmap="autumn", s=42)
        axR.axhline(0, color="0.85", lw=1, zorder=0)
        axR.set_xlim(-bmax, bmax); axR.set_ylim(-bmax * 0.55, bmax * 0.55)
        axR.set_aspect("equal")
        axR.set_xticks([]); axR.set_yticks([])
        axR.set_xlabel("body axis (head → tail)", fontsize=9)
        axR.set_ylabel("lateral", fontsize=9)
        axR.set_title("body frame, top-down  (tail-beat)", fontsize=11)

        tag = "   ⚠ HYDRO-LIFT LAUNCH (not swimming)" if launch[t] else ""
        fig.suptitle(f"{args.title}   |   t={t*dt:5.2f}s   speed={speed[t]:.2f} m/s"
                     f"   pitch={pitch[t]:+.0f}°{tag}",
                     fontsize=13, color="#c0392b" if launch[t] else "black")
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(frame)
        fig.clf()
    writer.close()
    print(f"[render_openwater] wrote {args.out}  ({T} frames @ {args.fps}fps)")


main()
