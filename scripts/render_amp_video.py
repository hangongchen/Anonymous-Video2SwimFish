#!/usr/bin/env python
"""Render AMP videos. No Isaac needed. Two modes:

  --mode rollout   (default, unchanged) a recorded rollout from play.py --record_demo
                   (npz: pc (T,N,3) world, root (T,13)) as a 3D fish inside the tank walls.

  --mode reference the AMP REFERENCE DATASET itself -- the motion the discriminator is trained to
                   call "real". Animates the stored feature Phi (NOT the raw video), because Phi is
                   exactly what D sees, so anything invisible here is invisible to D. Renders the OLD
                   and NEW datasets side by side.

Why the reference mode exists (2026-08-06): the old 44-dim Phi described only body SHAPE plus
world-frame speed magnitudes, so it could not express which way the fish was pointing relative to
where it was going. The new 46-dim Phi writes the velocity in the body's own (nose,left,up) basis.
The video makes that concrete: the OLD row can only draw the body, the NEW row can also draw the
travel-direction arrow, and on the real fish that arrow sits on the nose (median 10 deg off).
"""
from __future__ import annotations

import argparse
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



def quat_apply_inv(q, v):  # q (4,) wxyz, v (N,3) -> body frame
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return v @ R


def tank_edges(half, z0, z1):
    c = [(-half, -half), (half, -half), (half, half), (-half, half)]
    segs = []
    for i in range(4):
        a, b = c[i], c[(i+1) % 4]
        segs.append(([a[0], b[0]], [a[1], b[1]], [z0, z0]))   # bottom
        segs.append(([a[0], b[0]], [a[1], b[1]], [z1, z1]))   # top
        segs.append(([a[0], a[0]], [a[1], a[1]], [z0, z1]))   # verticals
    return segs


def midline_from_bend(lat, vert):
    """Rebuild a body midline (in body lengths) from the STORED bend profile.

    The stations are equally spaced by ARC LENGTH (both the reference builder and the env resample
    that way), so the along-body coordinate is not k/(K-1): it has to be integrated, taking out the
    part of each step that went sideways/vertically.  ds^2 = dx^2 + dlat^2 + dvert^2.
    Head is pinned at the origin. lat/vert: (T,K) -> along (T,K)."""
    K = lat.shape[1]
    ds = 1.0 / (K - 1)
    dl, dv = np.diff(lat, axis=1), np.diff(vert, axis=1)
    dx = np.sqrt(np.clip(ds ** 2 - dl ** 2 - dv ** 2, 0.0, None))
    return np.concatenate([np.zeros((len(lat), 1)), np.cumsum(dx, axis=1)], axis=1)


def load_reference(path, K):
    """Unpack an AMP reference npz into a common shape, tolerating both feature layouts."""
    d = np.load(path, allow_pickle=True)
    f = np.asarray(d["feat"], np.float64)
    names = [str(s) for s in d["vel_names"]] if "vel_names" in d else []
    nbend = 2 * K if f.shape[1] >= 2 * K + 1 and "n_bend_per_station" in d and int(
        d["n_bend_per_station"]) == 2 else K
    lat = f[:, :K]
    vert = f[:, K:2 * K] if nbend == 2 * K else np.zeros_like(lat)
    motion = f[:, nbend:]
    return dict(feat=f, lat=lat, vert=vert, motion=motion, names=names,
                seg=np.asarray(d["segment_id"]) if "segment_id" in d else np.zeros(len(f), np.int64),
                nbend=nbend, path=path)


def render_reference(args):
    """Side-by-side animation of the OLD and NEW reference datasets, drawn from Phi itself."""
    K = args.K
    old, new = load_reference(args.old, K), load_reference(args.new, K)
    for r in (old, new):
        r["along"] = midline_from_bend(r["lat"], r["vert"])
    # dir_nose / dir_left let us draw the TRAVEL DIRECTION in the body frame -- the whole point of
    # the new feature. The old feature has no such channel, which is exactly what the video shows.
    def _chan(r, nm):
        return r["motion"][:, r["names"].index(nm)] if nm in r["names"] else None
    new_dn, new_dl = _chan(new, "dir_nose"), _chan(new, "dir_left")

    T = max(len(old["feat"]), len(new["feat"]))
    T = min(T, args.max_frames) if args.max_frames > 0 else T
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=None)
    fig = plt.figure(figsize=(13.0, 7.4))
    rows = [("OLD  amp_reference_3d.npz  (44 dims: 40 bend + 4 world-frame motion)", old, None, None),
            ("NEW  amp_reference_3d_v2.npz  (46 dims: 40 bend + 6 BODY-frame motion)", new, new_dn, new_dl)]
    for t in range(T):
        for ri, (title, r, dn, dl) in enumerate(rows):
            n = len(r["feat"])
            i = t % n                                   # loop the shorter dataset rather than truncate
            newseg = i > 0 and r["seg"][i] != r["seg"][i - 1]
            # --- top-down body (lateral bend) ---
            ax = fig.add_subplot(2, 3, 3 * ri + 1)
            ax.plot(r["along"][i], r["lat"][i], "-", lw=3, color="#1f77b4" if ri else "#888888",
                    solid_capstyle="round")
            ax.scatter(r["along"][i], r["lat"][i], s=16,
                       c=np.linspace(0, 1, K), cmap="autumn_r", zorder=3)
            ax.scatter([0], [0], s=70, color="k", zorder=4)          # head marker
            if dn is not None:
                # travel direction in the BODY frame: +x is the nose, +y is body-left.
                ax.arrow(0.0, 0.0, 0.30 * dn[i], 0.30 * dl[i], width=0.008, color="#d62728",
                         length_includes_head=True, zorder=5)
                ax.text(0.02, -0.40, f"travel {np.degrees(np.arccos(np.clip(dn[i],-1,1))):5.1f}$\\degree$ "
                                     f"off the nose", fontsize=9, color="#d62728")
            else:
                ax.text(0.02, -0.40, "travel direction: NOT IN THE FEATURE", fontsize=9, color="#888888")
            ax.set_xlim(-0.35, 1.05); ax.set_ylim(-0.45, 0.45)
            ax.set_xlabel("along body (BL)"); ax.set_ylabel("left / right (BL)")
            ax.set_title(f"{title}\ntop view  |  frame {i}/{n}" + ("   [NEW SEGMENT]" if newseg else ""),
                         fontsize=9, loc="left")
            ax.grid(alpha=0.25)
            # --- side view (vertical bend) ---
            ax2 = fig.add_subplot(2, 3, 3 * ri + 2)
            if r["nbend"] == 2 * K:
                ax2.plot(r["along"][i], r["vert"][i], "-", lw=3, color="#2ca02c")
                ax2.scatter(r["along"][i], r["vert"][i], s=14, c=np.linspace(0, 1, K), cmap="autumn_r")
            else:
                ax2.text(0.35, 0.0, "not measured", ha="center", color="#888888")
            ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.45, 0.45)
            ax2.set_xlabel("along body (BL)"); ax2.set_ylabel("up / down (BL)")
            ax2.set_title("side view", fontsize=9, loc="left"); ax2.grid(alpha=0.25)
            # --- live motion-channel traces ---
            ax3 = fig.add_subplot(2, 3, 3 * ri + 3)
            w = int(args.trace_window * args.fps)
            a, b = max(0, i - w), min(n, i + w)
            for j, nm in enumerate(r["names"]):
                ax3.plot(np.arange(a, b), r["motion"][a:b, j], lw=1.2, label=nm)
            ax3.axvline(i, color="k", lw=1.0, alpha=0.6)
            ax3.set_xlim(a, max(b, a + 1)); ax3.set_ylim(-4.5, 7.5)
            ax3.legend(fontsize=7, ncol=2, loc="upper right")
            ax3.set_title("motion channels (what D sees besides shape)", fontsize=9, loc="left")
            ax3.grid(alpha=0.25)
        fig.suptitle("AMP reference dataset — the motion the discriminator is trained to call REAL",
                     fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(frame)
        fig.clf()
    writer.close()
    print(f"[render_amp_video] wrote {args.out}  ({T} frames @ {args.fps}fps)")
    print(f"  OLD {args.old}: {old['feat'].shape} motion={old['names'] or 'unnamed'} "
          f"segments={len(np.unique(old['seg']))}")
    print(f"  NEW {args.new}: {new['feat'].shape} motion={new['names']} "
          f"segments={len(np.unique(new['seg']))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rollout", "reference"], default="rollout")
    ap.add_argument("--demo", help="rollout mode: play.py --record_demo npz")
    ap.add_argument("--out", default=_P("${FISH_ROOT}/demo_out/zef_amp_ref/amp_deploy.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--half", type=float, default=1.0)      # tank half-size (m)
    ap.add_argument("--wall-z", type=float, default=1.0)
    ap.add_argument("--wall-h", type=float, default=2.0)
    ap.add_argument("--stride", type=int, default=1)
    # reference mode
    ap.add_argument("--old", default=_P("${FISH_ROOT}/demo_out/zef05_amp_ref/amp_reference_3d.npz"))
    ap.add_argument("--new", default=_P("${FISH_ROOT}/demo_out/zef05_amp_ref/amp_reference_3d_v2.npz"))
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--trace-window", type=float, default=4.0, help="seconds of motion trace shown")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    if args.mode == "reference":
        return render_reference(args)
    if not args.demo:
        ap.error("--demo is required for --mode rollout")

    d = np.load(args.demo)
    pc = np.asarray(d["pc"], np.float64)[::args.stride]      # (T,N,3) world
    root = np.asarray(d["root"], np.float64)[::args.stride]
    T = len(pc)
    z0, z1 = args.wall_z - args.wall_h/2, args.wall_z + args.wall_h/2
    segs = tank_edges(args.half, z0, z1)
    trail = root[:, 0:3]

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=None)
    fig = plt.figure(figsize=(6.4, 6.0))
    for t in range(T):
        ax = fig.add_subplot(111, projection="3d")
        for xs, ys, zs in segs:
            ax.plot(xs, ys, zs, color="#3a6ea5", lw=1.0, alpha=0.5)
        # head/tail color from body-frame x
        body = quat_apply_inv(root[t, 3:7], pc[t] - root[t, 0:3])
        col = body[:, 0]
        ax.scatter(pc[t, :, 0], pc[t, :, 1], pc[t, :, 2], c=col, cmap="autumn", s=14, depthshade=True)
        ax.plot(trail[:t+1, 0], trail[:t+1, 1], trail[:t+1, 2], color="0.4", lw=1.0, alpha=0.7)
        ax.set_xlim(-args.half, args.half); ax.set_ylim(-args.half, args.half); ax.set_zlim(z0, z1)
        ax.set_box_aspect((1, 1, args.wall_h/(2*args.half)))
        ax.view_init(elev=28, azim=-60 + 0.15*t)   # slow orbit
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"AMP tank swim  |  step {t*args.stride}", fontsize=10)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        writer.append_data(frame)
        fig.clf()
    writer.close()
    print(f"[render_amp_video] wrote {args.out}  ({T} frames @ {args.fps}fps, N={pc.shape[1]} pts)")


if __name__ == "__main__":
    main()
