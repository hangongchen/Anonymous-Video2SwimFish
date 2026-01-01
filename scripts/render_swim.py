"""Render swim .npz logs to an mp4 + trajectory plot using matplotlib (no Kit).

Usage:
  python scripts/render_swim.py --aniso demo_out/run_aniso.npz --iso demo_out/run_isotropic.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)



def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    pos = d["pos"]                       # (T, nb, 3)
    axis = unit(d["axis"])
    c0 = d["c0"]
    order = d["order"].astype(int)
    up = np.array([0.0, 0.0, 1.0])
    side = unit(np.cross(axis, up))
    if np.linalg.norm(np.cross(axis, up)) < 1e-3:
        side = np.array([0.0, 1.0, 0.0])
    rel = pos - c0[None, None, :]        # (T, nb, 3)
    along = rel @ axis                   # (T, nb)
    lateral = rel @ side
    vert = rel @ up
    soft_along = soft_lateral = None
    if "soft_pos" in d.files:
        sp = d["soft_pos"]               # (T, n_soft, 3)
        srel = sp - c0[None, None, :]
        soft_along = srel @ axis
        soft_lateral = srel @ side
    return dict(pos=pos, along=along[:, order], lateral=lateral[:, order],
                vert=vert[:, order], fwd=d["fwd"], dt=float(d["dt"]),
                fish_len=float(d["fish_len"]), freq=float(d["freq"]),
                mode=str(d["mode"]), order=order,
                soft_along=soft_along, soft_lateral=soft_lateral)


def frame_limits(runs):
    al = np.concatenate([r["along"].ravel() for r in runs])
    la = np.concatenate([r["lateral"].ravel() for r in runs])
    pad_a = 0.1 * (al.max() - al.min() + 1e-3)
    pad_l = 0.1 * (la.max() - la.min() + 1e-3)
    return (al.min() - pad_a, al.max() + pad_a), (la.min() - pad_l, la.max() + pad_l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aniso", type=str, default=_P("${FISH_ROOT}/demo_out/run_aniso.npz"))
    ap.add_argument("--iso", type=str, default=_P("${FISH_ROOT}/demo_out/run_isotropic.npz"))
    ap.add_argument("--out", type=str, default=_P("${FISH_ROOT}/demo_out"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--video_seconds", type=float, default=0.0,
                    help="0 = real-time; else compress whole run into this many s")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = []
    labels = []
    if os.path.exists(args.aniso):
        runs.append(load(args.aniso)); labels.append("ANISOTROPIC drag")
    if os.path.exists(args.iso):
        runs.append(load(args.iso)); labels.append("ISOTROPIC drag (control)")
    if not runs:
        raise SystemExit("no npz files found")

    T = min(r["along"].shape[0] for r in runs)
    dt = runs[0]["dt"]
    xlim, ylim = frame_limits(runs)

    # frame subsampling
    target_frames = int((args.video_seconds if args.video_seconds > 0 else T * dt) * args.fps)
    step_idx = np.linspace(0, T - 1, max(target_frames, 2)).astype(int)

    mp4 = os.path.join(args.out, "swim_compare.mp4")
    writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)

    n = len(runs)
    bl = max(r["fish_len"] for r in runs)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.0 * n), squeeze=False)

    for fi in step_idx:
        for row, (r, lab) in enumerate(zip(runs, labels)):
            al = r["along"][fi]; la = r["lateral"][fi]
            cen_a = al.mean(); cen_l = la.mean()
            fwd = r["fwd"][fi]
            # --- lab frame: full corridor, forward progress visible ---
            axL = axes[row, 0]; axL.clear()
            axL.set_xlim(*xlim); axL.set_ylim(*ylim); axL.set_aspect("equal", adjustable="box")
            trail = slice(0, fi + 1, max(1, fi // 250 + 1))
            axL.plot(r["along"][trail].mean(1), r["lateral"][trail].mean(1), "-", color="0.7", lw=1.0)
            if r.get("soft_along") is not None:
                axL.scatter(r["soft_along"][fi], r["soft_lateral"][fi], s=3,
                            c="navajowhite", alpha=0.5, edgecolors="none", zorder=2)
            axL.plot(al, la, "-o", color="tab:orange", lw=3, ms=5, zorder=3)
            axL.plot(al[0], la[0], "o", color="red", ms=10, zorder=4)
            axL.set_title(f"{lab} — lab frame   t={fi*dt:4.1f}s   forward={fwd:+.2f} m ({fwd/r['fish_len']:+.1f} BL)",
                          fontsize=10)
            axL.set_xlabel("forward along body axis (m)"); axL.set_ylabel("lateral (m)")
            axL.grid(True, alpha=0.3)
            # --- body frame: centroid-subtracted, undulation shape visible ---
            axB = axes[row, 1]; axB.clear()
            axB.set_xlim(-0.7 * bl, 0.7 * bl); axB.set_ylim(-0.5 * bl, 0.5 * bl)
            axB.set_aspect("equal", adjustable="box")
            if r.get("soft_along") is not None:
                axB.scatter(r["soft_along"][fi] - cen_a, r["soft_lateral"][fi] - cen_l, s=5,
                            c="navajowhite", alpha=0.55, edgecolors="none", zorder=2,
                            label="soft body")
            axB.plot(al - cen_a, la - cen_l, "-o", color="tab:orange", lw=3, ms=6, zorder=3)
            axB.plot(al[0] - cen_a, la[0] - cen_l, "o", color="red", ms=11, label="head", zorder=4)
            axB.plot(al[-1] - cen_a, la[-1] - cen_l, "s", color="tab:blue", ms=8, label="tail", zorder=4)
            axB.set_title(f"{lab} — body frame (undulation)", fontsize=10)
            axB.set_xlabel("along (m)"); axB.set_ylabel("lateral (m)")
            axB.grid(True, alpha=0.3); axB.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)[..., :3]
        writer.append_data(np.ascontiguousarray(frame))
    writer.close()
    plt.close(fig)
    print("wrote", mp4)

    # static snapshot montage: lab-frame spine at several times for each run
    snaps = np.linspace(0, T - 1, 5).astype(int)
    figm, axm = plt.subplots(n, 1, figsize=(12, 3.0 * n), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for row, (r, lab) in enumerate(zip(runs, labels)):
        ax = axm[row, 0]
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal", adjustable="box")
        for si, fi in enumerate(snaps):
            col = cmap(si / (len(snaps) - 1))
            if r.get("soft_along") is not None:
                ax.scatter(r["soft_along"][fi], r["soft_lateral"][fi], s=3, color=col,
                           alpha=0.35, edgecolors="none", zorder=2)
            ax.plot(r["along"][fi], r["lateral"][fi], "-o", color=col, lw=2, ms=4,
                    label=f"t={fi*dt:.1f}s", zorder=3)
            ax.plot(r["along"][fi][0], r["lateral"][fi][0], "o", color=col, ms=9, mec="k", zorder=4)
        net = r["fwd"][-1] - r["fwd"][0]
        ax.set_title(f"{lab}: spine over time (dots=head)   net forward = {net:+.2f} m "
                     f"({net/r['fish_len']:+.1f} BL)", fontsize=11)
        ax.set_xlabel("forward along body axis (m)"); ax.set_ylabel("lateral (m)")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8, ncol=5)
    figm.tight_layout()
    montage = os.path.join(args.out, "snapshots.png")
    figm.savefig(montage, dpi=130); plt.close(figm)
    print("wrote", montage)

    # static trajectory comparison
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for r, lab in zip(runs, labels):
        tt = np.arange(r["fwd"].shape[0]) * r["dt"]
        ax2.plot(tt, r["fwd"], lw=2.5, label=f"{lab}  (net {r['fwd'][-1]-r['fwd'][0]:+.3f} m)")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("forward distance along body axis (m)")
    ax2.set_title("Net forward progress: anisotropic drag rectifies the body wave into thrust")
    ax2.grid(True, alpha=0.3); ax2.legend()
    fig2.tight_layout()
    traj = os.path.join(args.out, "trajectory_compare.png")
    fig2.savefig(traj, dpi=130)
    print("wrote", traj)


if __name__ == "__main__":
    main()
