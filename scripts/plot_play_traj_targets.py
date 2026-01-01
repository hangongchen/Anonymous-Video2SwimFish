"""Plot an env0 play trajectory WITH every historical target.

The play recorder (scripts/rl_games/play.py) saves /tmp/misty_traj_ep{N}.npy as a
(T, 6) array: columns [fish_x, fish_y, fish_z, target_x, target_y, target_z] per
control step, all env-LOCAL. In multi_target mode the target JUMPS whenever the fish
reaches it, so the distinct target rows ARE the history of goals. This script draws:
  - the fish path (blue, start=green, end=orange),
  - every distinct target as a numbered red star (order the fish was sent to them),
  - a dashed success-radius circle around each target,
  - a green tick at the exact step the fish reached a target (target changed).

Usage:  python scripts/plot_play_traj_targets.py /tmp/misty_traj_ep1.npy [--radius 0.4]
        (no path -> auto-pick the /tmp/misty_traj_ep*.npy with the MOST reaches)
"""
import argparse
import glob
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("path", nargs="?", default=None)
parser.add_argument("--radius", type=float, default=0.4, help="success radius to draw around each target")
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()


def distinct_targets(fish, tg, radius, tol=1e-3):
    """Return (targets[K,3], reach_step_indices) — a new target each time it moves, but only
    counting it as a genuine REACH if the fish was actually inside `radius` of the OLD target
    at the moment it changed. This rejects the episode's final reset frame, where
    target_positions_w is already the NEXT episode's (far-away) target and the fish has just
    snapped back to spawn (a change with the fish NOWHERE near the old target)."""
    tgs, reach_idx = [tg[0]], []
    reach_tol = max(radius * 1.5, radius + 0.1)      # generous: reach detection, not scoring
    for i in range(1, len(tg)):
        if np.linalg.norm(tg[i] - tgs[-1]) > tol:
            d_old = np.linalg.norm(fish[i - 1] - tgs[-1])   # fish vs the target it just left
            if d_old <= reach_tol:                          # genuine reach -> keep + count
                tgs.append(tg[i]); reach_idx.append(i)
            # else: reset artifact -> ignore this target change entirely
    return np.asarray(tgs), reach_idx


def pick_best():
    best, best_n = None, -1
    for p in sorted(glob.glob("/tmp/misty_traj_ep*.npy")):
        d = np.load(p)
        n = len(distinct_targets(d[:, 0:3], d[:, 3:6], args.radius)[1])   # genuine reaches
        if n > best_n:
            best, best_n = p, n
    return best


path = args.path or pick_best()
if path is None:
    raise SystemExit("no /tmp/misty_traj_ep*.npy found — run a play first")

data = np.load(path)
fish, tg = data[:, 0:3], data[:, 3:6]
targets, reach_idx = distinct_targets(fish, tg, args.radius)
# closest approach to the FIRST target (the one it had for the whole no-reach episode)
_min_first = float(np.linalg.norm(fish - targets[0], axis=1).min())
print(f"[plot] closest approach to target 1: {_min_first:.3f} m (radius {args.radius})")
n_reach = len(reach_idx)
print(f"[plot] {path}: {len(data)} steps, {len(targets)} distinct targets, {n_reach} reaches")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

# fish path, colored by time (light -> dark = start -> end)
t = np.linspace(0, 1, len(fish))
ax.scatter(fish[:, 0], fish[:, 1], fish[:, 2], c=t, cmap="Blues", s=4, alpha=0.6)
ax.plot(fish[:, 0], fish[:, 1], fish[:, 2], c="#4a90d9", lw=0.8, alpha=0.5, label="fish path")
ax.scatter(*fish[0], c="green", s=90, label="start", depthshade=False)
ax.scatter(*fish[-1], c="orange", s=90, label="end", depthshade=False)

# every historical target, numbered in visit order + a dashed reach circle
th = np.linspace(0, 2 * np.pi, 60)
for k, tgt in enumerate(targets):
    reached = k < n_reach          # targets[0..n_reach-1] were reached; the last one usually was NOT
    col = "#c0392b" if reached else "#e67e22"
    _lab = "target (reached)" if reached else "target (not reached)"
    _seen = globals().setdefault("_legseen", set())   # one legend entry per label
    if _lab in _seen:
        _lab = None
    else:
        _seen.add(_lab)
    ax.scatter(*tgt, marker="*", s=280, c=col, edgecolors="k", linewidths=0.6,
               label=_lab, depthshade=False)
    ax.text(tgt[0], tgt[1], tgt[2] + 0.05, f"{k+1}", fontsize=11, fontweight="bold", ha="center")
    # success-radius ring (drawn in the target's z-plane, horizontal)
    ax.plot(tgt[0] + args.radius * np.cos(th), tgt[1] + args.radius * np.sin(th),
            np.full_like(th, tgt[2]), c=col, lw=0.8, ls="--", alpha=0.5)

# mark the exact reach points on the path
for i in reach_idx:
    ax.scatter(*fish[i], marker="X", s=110, c="lime", edgecolors="k", linewidths=0.7,
               depthshade=False, zorder=6)

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title(f"env0 play — {len(targets)} targets given, {n_reach} reached  (green X = reach point)\n{os.path.basename(path)}")
ax.legend(loc="upper left", fontsize=9)
try:
    ax.set_box_aspect((1, 1, 0.6))
except Exception:  # noqa: BLE001
    pass

out = args.out or (_P("${FISH_ROOT}/demo_out/") + os.path.basename(path).replace(".npy", "_targets.png"))
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=120)
print(f"[plot] SAVED {out}")
