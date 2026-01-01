#!/usr/bin/env python
"""Apply the SAME 3D body-line measurement (left/right + up/down per point, in the body's own frame,
divided by the TRUE 3D length) to the SIM fish, and compare to the real fish from Phase 3.

Sim fish nodes are already 3D, so no cameras -- just bin along the body, take per-station 3D centroids,
then the same body_frame_feature. Head/tail fixed: the sim nose is at HIGH along (fwd = body -X)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = 20


def rotmat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def body_frame_feature(M3d):
    head = M3d[0]
    rel = M3d - head
    fwd = rel[max(1, K // 4)] - rel[0]
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    up = np.array([0.0, 0.0, 1.0]); up = up - up.dot(fwd) * fwd; up /= np.linalg.norm(up) + 1e-9
    left = np.cross(up, fwd)
    body_len = np.linalg.norm(np.diff(M3d, axis=0), axis=1).sum()
    return rel @ fwd / body_len, rel @ left / body_len, rel @ up / body_len, body_len


def sim_midline_3d(nodes, root):
    R = rotmat(root[3:7])
    fwd3 = R @ np.array([-1.0, 0, 0]); fwd3 /= np.linalg.norm(fwd3) + 1e-9   # nose dir, full 3D
    along = (nodes - root[0:3]) @ fwd3
    lo, hi = np.percentile(along, 1), np.percentile(along, 99)
    edges = np.linspace(lo, hi, K + 1)
    pts = []
    for i in range(K):
        sel = (along >= edges[i]) & (along <= edges[i + 1])
        pts.append(nodes[sel].mean(0) if sel.sum() > 2 else (np.nan, np.nan, np.nan))
    pts = np.array(pts)
    pts = pts[::-1]                       # reverse: head (nose, high along) -> tail
    good = ~np.isnan(pts[:, 0])
    if good.sum() < 12:
        return None
    idx = np.where(good)[0]
    pts = np.stack([np.interp(np.arange(K), idx, pts[good, k]) for k in range(3)], 1)
    return pts


d = np.load("/tmp/amp09_ep100_det.npz")
pc = np.asarray(d["pc"], float); root = np.asarray(d["root"], float)
lat, ver, blens, curves = [], [], [], {}
show = [60, 120, 200, 300]
for t in range(30, min(len(root), 350)):
    M = sim_midline_3d(pc[t], root[t])
    if M is None:
        continue
    al, la, ve, bl = body_frame_feature(M)
    lat.append(la); ver.append(ve); blens.append(bl)
    if t in show:
        curves[t] = (M, al, la, ve)
lat = np.array(lat); ver = np.array(ver); blens = np.array(blens)
BL = 0.5
print(f"[sim3d] {len(lat)} frames; body length {blens.mean()/BL:.3f}+-{blens.std()/BL:.3f} BL "
      f"({blens.mean():.3f} m)")
sim_lr = np.abs(lat).mean(0) * np.sqrt(2); sim_ud = np.abs(ver).mean(0) * np.sqrt(2)
print(f"[sim3d] tail amplitude: left/right {sim_lr[-1]:.3f} BL  up/down {sim_ud[-1]:.3f} BL  "
      f"(up-down / left-right = {sim_ud[-1]/max(sim_lr[-1],1e-6):.2f})")

# real fish from phase3
ref = np.load("demo_out/zef05_amp_ref/bend3d_raw.npz")
ref_lr = np.abs(ref["lateral"]).mean(0) * np.sqrt(2); ref_ud = np.abs(ref["vertical"]).mean(0) * np.sqrt(2)
print(f"[real ] tail amplitude: left/right {ref_lr[-1]:.3f} BL  up/down {ref_ud[-1]:.3f} BL  "
      f"(up-down / left-right = {ref_ud[-1]/max(ref_lr[-1],1e-6):.2f})")

s = np.linspace(0, 1, K)
fig = plt.figure(figsize=(15, 9))
fig.suptitle("SIM fish 3D body line vs REAL fish — left/right AND up/down (body's own frame)", fontsize=13)
ax = fig.add_subplot(2, 2, 1)
for t, (M, al, la, ve) in curves.items():
    ax.plot(al, la, "-o", ms=3, lw=1.5, label=f"t{t}")
ax.axhline(0, color="k", lw=0.4, ls=":"); ax.set_title("(a) SIM top view: LEFT/RIGHT bend")
ax.set_xlabel("along body (head→tail)"); ax.set_ylabel("left/right (BL)"); ax.legend(fontsize=7)
ax = fig.add_subplot(2, 2, 2)
for t, (M, al, la, ve) in curves.items():
    ax.plot(al, ve, "-o", ms=3, lw=1.5)
ax.axhline(0, color="k", lw=0.4, ls=":"); ax.set_title("(b) SIM side view: UP/DOWN bend")
ax.set_xlabel("along body (head→tail)"); ax.set_ylabel("up/down (BL)")
ax = fig.add_subplot(2, 2, 3)
ax.plot(s, sim_lr, "b-o", ms=3, lw=2, label="SIM left/right")
ax.plot(s, sim_ud, "r-o", ms=3, lw=2, label="SIM up/down")
ax.plot(s, ref_lr, "b--s", ms=3, lw=1.5, alpha=0.6, label="REAL left/right")
ax.plot(s, ref_ud, "r--s", ms=3, lw=1.5, alpha=0.6, label="REAL up/down")
ax.set_title("(c) how big each bend is: SIM (solid) vs REAL (dashed)")
ax.set_xlabel("body station (head→tail)"); ax.set_ylabel("amplitude (BL)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = fig.add_subplot(2, 2, 4, projection="3d")
for t, (M, al, la, ve) in curves.items():
    Mc = (M - M[0]) / BL
    ax.plot(Mc[:, 0], Mc[:, 1], Mc[:, 2], "-o", ms=2)
ax.set_title("(d) the SIM 3D body lines (BL)"); ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("demo_out/sim_bend3d.png", dpi=110)
print("[sim3d] saved demo_out/sim_bend3d.png")
