#!/usr/bin/env python
"""Visualize a DETERMINISTIC swim rollout (play.py --record_demo npz): trajectory + gait + undulation.

Produces a multi-panel PNG from the recorded FEM cloud + root pose + joints (no Isaac / camera needed,
so it sidesteps the FEM-Replicator stall). Panels:
  (a) top-down root trajectory, colored by time
  (b) live lateral-DOF joint angles vs time  (the traveling wave / gait)
  (c) 6 top-view body midline snapshots (FEM cloud, heading-aligned) across one span -> the undulation
  (d) tail-tip lateral offset vs time (the tail beat) + its FFT peak frequency

  python scripts/viz_swim_rollout.py <rollout.npz> --out demo_out/swim_viz.png --bl 0.5
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--out", default="demo_out/swim_viz.png")
ap.add_argument("--bl", type=float, default=0.5)
ap.add_argument("--title", default="")
args = ap.parse_args()

d = np.load(args.npz)
pc = np.asarray(d["pc"], float)          # (T,N,3) world (env0-local)
root = np.asarray(d["root"], float)      # (T,13)
jp = np.asarray(d["joint_pos"], float)   # (T,J)
dt = float(d["step_dt"]); T = len(root); BL = args.bl
pos = root[:, 0:3]; quat = root[:, 3:7]
tvec = np.arange(T) * dt


def quat_inv_rot(q, v):                   # world->body (per frame), v (N,3)
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return v @ R                          # == R^T applied per row (inverse)


# body-frame MIDLINE per frame: bin nodes by along-body position, take MEAN lateral per bin (averages
# out the body WIDTH so we get a clean centerline, not the full 165-node cloud).
NB = 16
mid_s, mid_y = [], []
for t in range(T):
    b = quat_inv_rot(quat[t], pc[t] - pos[t])
    a = b[:, 0]
    edges = np.linspace(a.min(), a.max(), NB + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    lat = np.array([b[(a >= edges[i]) & (a <= edges[i+1]), 1].mean() if np.any((a >= edges[i]) & (a <= edges[i+1]))
                    else np.nan for i in range(NB)])
    mid_s.append(ctr); mid_y.append(lat)
mid_s = np.array(mid_s); mid_y = np.array(mid_y)
tail_lat = mid_y[:, -1]                                  # tail-station lateral (body y), midline
tail_ac = tail_lat - tail_lat.mean()                    # DC-removed tail beat (the oscillation only)

live = np.arange(jp.shape[1]) % 3 == 2
fig = plt.figure(figsize=(15, 9))
sub = args.title or args.npz.split("/")[-1]
fig.suptitle(f"Deterministic swim rollout — {sub}   ({T*dt:.0f}s @ {1/dt:.0f}Hz)", fontsize=13)

# (a) top-down trajectory
ax = fig.add_subplot(2, 2, 1)
sc = ax.scatter(pos[:, 0], pos[:, 1], c=tvec, cmap="viridis", s=6)
ax.plot(pos[0, 0], pos[0, 1], "go", ms=9, label="start"); ax.plot(pos[-1, 0], pos[-1, 1], "rs", ms=9, label="end")
path = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()
net = np.linalg.norm(pos[-1, :2] - pos[0, :2])
ax.set_title(f"(a) top-down path: {path/BL:.1f} BL traveled, {net/BL:.2f} BL net"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.axis("equal"); ax.legend(fontsize=8); plt.colorbar(sc, ax=ax, label="t (s)")

# (b) live-DOF joint angles (traveling wave)
ax = fig.add_subplot(2, 2, 2)
liveidx = np.where(live)[0]
for i, j in enumerate(liveidx):
    ax.plot(tvec, jp[:, j] + 0.0, lw=0.8, color=plt.cm.plasma(i / max(1, len(liveidx)-1)))
ax.set_title(f"(b) lateral-DOF joint angles (head→tail = purple→yellow)"); ax.set_xlabel("t (s)"); ax.set_ylabel("angle (rad)")
ax.set_xlim(0, min(T*dt, 8))

# (c) midline undulation snapshots (clean centerline, ~1.5 s window)
ax = fig.add_subplot(2, 2, 3)
idxs = np.linspace(T//4, T//4 + int(1.5/dt), 6).astype(int)
idxs = idxs[idxs < T]
for i, t in enumerate(idxs):
    ax.plot(mid_s[t] / BL, mid_y[t] / BL, "-o", ms=3, color=plt.cm.cool(i/max(1,len(idxs)-1)), lw=1.6, alpha=0.9)
ax.set_title("(c) body MIDLINE (heading-aligned) over ~1.5 s → undulation"); ax.set_xlabel("along body (BL)"); ax.set_ylabel("lateral (BL)")
ax.axhline(0, color="k", lw=0.5, ls=":"); ax.axis("equal")

# (d) tail beat (DC-removed) + frequency
ax = fig.add_subplot(2, 2, 4)
ax.plot(tvec, tail_ac / BL, "b-", lw=1)
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_title("(d) tail-station lateral beat (mean-removed)"); ax.set_xlabel("t (s)"); ax.set_ylabel("tail lateral (BL)")
if T > 16:
    f = np.fft.rfftfreq(T, dt); A = np.abs(np.fft.rfft(tail_ac))
    fbeat = f[1:][np.argmax(A[1:])]
    bias = tail_lat.mean() / BL
    ax.text(0.98, 0.02, f"beat ≈ {fbeat:.2f} Hz\nbeat amp {np.abs(tail_ac).mean()/BL:.3f} BL\nDC bias {bias:+.2f} BL (curve)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))
plt.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.savefig(args.out, dpi=110)
print(f"[viz] saved {args.out}")
jvel = np.abs(np.diff(jp[:, live], axis=0) / dt)
print(f"[viz] path {path/BL:.2f} BL, net {net/BL:.2f} BL, tail-beat amp {np.abs(tail_ac).mean()/BL:.3f} BL "
      f"(DC bias {tail_lat.mean()/BL:+.2f} BL), live-DOF jvel median {np.median(jvel):.3f} rad/s")
