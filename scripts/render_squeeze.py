#!/usr/bin/env python
"""Render the gripper-squeeze experiment clearly: a 2-panel animation
  (left)  TOP-DOWN  (x horizontal, y vertical): the two jaw plates close in y
  (right) HEAD-ON   (y horizontal, z vertical) at the MID-BODY: the flesh pinches
plus a diagnostic PNG (commanded jaw closure + ACTUAL mid-body compression + velocities).

The key correction vs the first version: we report ACTUAL tissue compression (y-width of the nodes
BETWEEN the jaws), not commanded jaw travel -- the soft mesh lets the jaws partially tunnel in."""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="demo_out/gripper_squeeze/squeeze_log.npz")
ap.add_argument("--plot", default="demo_out/gripper_squeeze/squeeze_diagnostic.png")
ap.add_argument("--video", default="demo_out/gripper_squeeze/squeeze.mp4")
ap.add_argument("--fps", type=int, default=25)
args = ap.parse_args()

d = np.load(args.npz)
t, nv, jv = d["t"], d["nodal_vmax"], d["joint_vmax"]
FT, FP, FJ = d["frames_t"], d["frames_pts"], d["frames_jaw"]
jd = d["jaw_dims"]; cen = d["center"]; cx = float(cen[0]); hw = float(d["half_width_y"])
JVEL_LIMIT, NVEL_LIMIT = 200.0, 50.0
halfx = jd[0] / 2.0                    # jaw x half-span -> "between the jaws" mask
midmask = np.abs(FP[:, :, 0] - cx) < halfx

# actual mid-body y-width per recorded frame, and inner jaw half-gap
mid_yw = np.array([FP[k][midmask[k], 1].ptp() for k in range(len(FT))])
inner = np.array([np.abs(FJ[k][:, 1]).min() - jd[1] / 2.0 for k in range(len(FT))]) * 2.0  # full inner gap
mid_yw0 = mid_yw[:5].mean()
compress = (mid_yw0 - mid_yw) / mid_yw0 * 100.0    # ACTUAL tissue compression %

# blow-up index in the fine per-step log
mask = (nv > NVEL_LIMIT) | (jv > JVEL_LIMIT)
blow_i = int(np.argmax(mask)) if mask.any() else None
tb = t[blow_i] if blow_i is not None else 1e9

# ---------------- diagnostic PNG ----------------
fig, ax1 = plt.subplots(figsize=(9, 4.8))
ax1.plot(FT, inner * 1000, color="0.55", lw=2, label="jaw inner gap (mm)")
ax1.plot(FT, mid_yw * 1000, color="#16a085", lw=2, label="mid-body width between jaws (mm)")
ax1.axhline(hw * 2 * 1000, color="#16a085", ls=":", lw=1, alpha=0.6)
ax1.set_xlabel("time (s)"); ax1.set_ylabel("width / gap (mm)"); ax1.set_ylim(0, 300)
ax2 = ax1.twinx()
ax2.plot(t, jv, color="#c0392b", lw=1.3, label="max |joint vel| (rad/s)")
ax2.plot(t, nv, color="#2e86de", lw=1.0, alpha=0.8, label="max |nodal vel| (m/s)")
ax2.axhline(JVEL_LIMIT, color="#c0392b", ls=":", lw=1, alpha=0.7)
ax2.set_ylabel("velocity (joint rad/s, nodal m/s)"); ax2.set_yscale("log"); ax2.set_ylim(1e-2, 1e3)
if blow_i is not None:
    cb = np.interp(tb, FT, compress)
    ax1.axvline(tb, color="k", ls="--", lw=1.2)
    ax1.annotate(f"BLOW-UP\nactual compression ~{cb:.0f}%\nt={tb:.2f}s", (tb, 250),
                 xytext=(tb - 1.9, 275), fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", color="k"))
ax1.set_title("FEM fish squeezed by a gripper (dt=1/960, youngs=1e5, elasticityDamping=0.05)")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)
fig.tight_layout(); fig.savefig(args.plot, dpi=130); plt.close(fig)
print(f"[render_squeeze] wrote {args.plot}  (peak actual compression {compress.max():.0f}%)")

# ---------------- 2-panel video ----------------
allp = FP.reshape(-1, 3)
xlim = (allp[:, 0].min() - 0.03, allp[:, 0].max() + 0.03)
ymax = float(np.abs(FJ[0][:, 1]).max()) + jd[1] + 0.01
zlim = (min(0.0, allp[:, 2].min()) - 0.01, allp[:, 2].max() + 0.05)
writer = imageio.get_writer(args.video, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
fig = plt.figure(figsize=(11, 5.2))
for k in range(len(FT)):
    tk = FT[k]; pts = FP[k]; jaw = FJ[k]; blown = tk >= tb
    ya = float(jaw[:, 1].max()); yb = float(jaw[:, 1].min())     # jaw centers +y / -y
    dy = jd[1] / 2.0
    mid = pts[midmask[k]]

    # -- top-down (x,y) --
    axL = fig.add_subplot(1, 2, 1)
    axL.scatter(pts[:, 0], pts[:, 1], c=pts[:, 0], cmap="autumn", s=10)
    axL.add_patch(Rectangle((xlim[0], ya - dy), xlim[1] - xlim[0], 2 * dy, color="#c0392b", alpha=0.55))
    axL.add_patch(Rectangle((xlim[0], yb - dy), xlim[1] - xlim[0], 2 * dy, color="#2e86de", alpha=0.55))
    axL.set_xlim(*xlim); axL.set_ylim(-ymax, ymax); axL.set_aspect("equal")
    axL.set_title("TOP-DOWN  (jaws close in y)", fontsize=10)
    axL.set_xlabel("x  head→tail"); axL.set_ylabel("y  (squeeze axis)")

    # -- head-on cross-section (y,z) of the mid-body nodes --
    axR = fig.add_subplot(1, 2, 2)
    if len(mid):
        axR.scatter(mid[:, 1], mid[:, 2], c=mid[:, 0], cmap="viridis", s=22)
    axR.add_patch(Rectangle((ya - dy, zlim[0]), 2 * dy, zlim[1] - zlim[0], color="#c0392b", alpha=0.55))
    axR.add_patch(Rectangle((yb - dy, zlim[0]), 2 * dy, zlim[1] - zlim[0], color="#2e86de", alpha=0.55))
    axR.set_xlim(-ymax, ymax); axR.set_ylim(*zlim); axR.set_aspect("equal")
    axR.set_title("HEAD-ON cross-section (mid-body)", fontsize=10)
    axR.set_xlabel("y  (squeeze axis)"); axR.set_ylabel("z  (up)")

    fig.suptitle(f"gripper squeeze  |  t={tk:4.2f}s   inner gap {inner[k]*1000:5.1f} mm   "
                 f"tissue compression {compress[k]:4.0f}%" + ("   <<< BLEW UP" if blown else ""),
                 fontsize=12, color=("#c0392b" if blown else "k"))
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
        fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
    writer.append_data(frame); fig.clf()
writer.close()
print(f"[render_squeeze] wrote {args.video}  ({len(FT)} frames)")
