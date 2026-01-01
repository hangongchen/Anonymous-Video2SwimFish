#!/usr/bin/env python
"""Visualize ONE agent-fish midline the way the discriminator builds it, from the 165 FEM mesh nodes.

Parallels the ZeF per-view midline viz: LEFT = horizontal plane (along x lateral) = exactly what the
disc sees today (top-view equivalent) with the extracted midline; RIGHT = vertical plane (along x z) =
the agent's 3D body shape the disc currently DISCARDS (side-view equivalent). Same head/tail markers.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_amp_features import anterior_aligned_profile

ap = argparse.ArgumentParser()
ap.add_argument("--demo", default="/tmp/amp09_ep100_det.npz")
ap.add_argument("--frame", type=int, default=-1, help="-1 = auto-pick a well-bent frame")
ap.add_argument("--out", default="demo_out/agent_midline_viz.png")
ap.add_argument("--K", type=int, default=20)
args = ap.parse_args()


def rotmat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def frame_coords(pc, root):
    R = rotmat(root[3:7])
    fwd = R @ np.array([-1.0, 0, 0]); fwd[2] = 0; fwd /= np.linalg.norm(fwd[:2]) + 1e-9   # nose, horizontal
    perp = np.array([-fwd[1], fwd[0], 0.0])                                                # left, horizontal
    rel = pc - root[0:3]
    along = rel @ fwd                          # + toward the nose
    lateral = rel @ perp                       # + to the left
    vert = rel[:, 2]                            # world z (up) -- the discarded axis
    return along, lateral, vert


def binned_midline(along, off, nb=18):
    lo, hi = np.percentile(along, 1), np.percentile(along, 99)
    edges = np.linspace(lo, hi, nb + 1)
    a, o = [], []
    for i in range(nb):
        sel = (along >= edges[i]) & (along <= edges[i + 1])
        if sel.sum() > 2:
            a.append(0.5 * (edges[i] + edges[i + 1])); o.append(off[sel].mean())
    return np.array(a), np.array(o)


d = np.load(args.demo)
pc = np.asarray(d["pc"], float); root = np.asarray(d["root"], float); T = len(root)
# auto-pick a frame with a clear lateral bend (large spread of the horizontal midline)
if args.frame < 0:
    best, bestv = T // 3, -1
    for t in range(30, min(T, 350)):
        al, la, _ = frame_coords(pc[t], root[t])
        _, mo = binned_midline(al, la)
        v = mo.max() - mo.min() if len(mo) else 0
        if v > bestv:
            bestv, best = v, t
    fr = best
else:
    fr = args.frame
al, la, ve = frame_coords(pc[fr], root[fr])
BL = 0.5
# head = nose end (max along); order stations head->tail
am_h, off_h = binned_midline(al, la)          # horizontal midline
am_v, off_v = binned_midline(al, ve)          # vertical midline
# faithful disc profile (K offsets) for the amplitude annotation
prof = anterior_aligned_profile(al, la, args.K)

fig, axs = plt.subplots(1, 2, figsize=(14, 5.2))
fig.suptitle(f"AGENT fish midline from the 165 FEM nodes  (frame {fr})  — how the discriminator builds it",
             fontsize=13)
for ax, (off, am, off_arr, name, note) in zip(axs, [
        (la, am_h, off_h, "HORIZONTAL plane (along × lateral)", "= what the discriminator SEES today (top-view equiv.)"),
        (ve, am_v, off_v, "VERTICAL plane (along × z-up)",       "= the 3D shape the disc DISCARDS (side-view equiv.)")]):
    ax.scatter(al / BL, off / BL, s=10, c="0.6", alpha=0.5, label="165 FEM nodes")
    if len(am):
        ax.plot(am / BL, off_arr / BL, "-o", color="lime", ms=5, lw=2.2, label="extracted midline")
        ax.plot(am[0] / BL, off_arr[0] / BL, "s", color="red", ms=11, label="head (nose)")
        ax.plot(am[-1] / BL, off_arr[-1] / BL, "^", color="cyan", ms=11, label="tail")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_title(f"{name}\n{note}", fontsize=10)
    ax.set_xlabel("along body (BL, + = nose)"); ax.set_ylabel("offset (BL)")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.3)
# match y-limits so the vertical vs horizontal amplitude is comparable
yl = max(abs(np.r_[la, ve]).max() / BL, 0.05)
for ax in axs:
    ax.set_ylim(-yl, yl)
plt.tight_layout(rect=[0, 0, 1, 0.94])
import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.savefig(args.out, dpi=115)
print(f"[agent-midline] saved {args.out}  frame {fr}")
print(f"[agent-midline] horizontal midline p2p {(off_h.max()-off_h.min())/BL:.3f} BL  |  "
      f"vertical midline p2p {(off_v.max()-off_v.min())/BL:.3f} BL  "
      f"(disc keeps horizontal, discards vertical)")
