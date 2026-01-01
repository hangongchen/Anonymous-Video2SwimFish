#!/usr/bin/env python
"""Visualize the AMP BEND PROFILE (the 20-dim bend part of Phi) for a rollout vs the ZeF05 reference.

The bend profile = the fish's midline lateral offset at K=20 stations head->tail, in the
world-horizontal swim plane, heading-aligned + anterior-tangent aligned (EXACTLY what _amp_feature
feeds the discriminator). This shows what "bend" the AMP disc actually sees.

  python scripts/viz_bend_profile.py --demo /tmp/amp09_ep100_det.npz --out demo_out/bend_profile_viz.png
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
ap.add_argument("--ref", default="demo_out/zef05_amp_ref/amp_reference.npz")
ap.add_argument("--env", default="demo_out/zef05_amp_ref/bend_envelope.npy")
ap.add_argument("--out", default="demo_out/bend_profile_viz.png")
ap.add_argument("--K", type=int, default=20)
args = ap.parse_args()
K = args.K


def rotmat(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


d = np.load(args.demo)
pc = np.asarray(d["pc"], float)          # (T,N,3) world
root = np.asarray(d["root"], float)      # (T,13)
dt = float(d["step_dt"]); T = len(root)
# per-frame bend profile, replicating _amp_feature exactly
prof = np.zeros((T, K))
for t in range(T):
    R = rotmat(root[t, 3:7])
    fwd = R @ np.array([-1.0, 0, 0]); fwd[2] = 0; fwd /= np.linalg.norm(fwd[:2]) + 1e-9
    perp = np.array([-fwd[1], fwd[0], 0.0])
    rel = pc[t] - root[t, 0:3]
    along = rel @ fwd
    lateral = rel @ perp
    prof[t] = anterior_aligned_profile(along, lateral, K)

ref = np.load(args.ref)["feat"][:, :K]   # (Tr, K) reference bend profiles
env = np.load(args.env)                   # (K,) reference amplitude envelope
s = np.linspace(0, 1, K)                  # head(0) -> tail(1)

fig = plt.figure(figsize=(15, 9))
fig.suptitle("AMP bend profile  (midline lateral offset at K=20 stations, head→tail; what the discriminator sees)", fontsize=13)

# (a) fish space-time heatmap: station x time -> the TRAVELING WAVE
ax = fig.add_subplot(2, 2, 1)
w = min(T, int(6 / dt))
im = ax.imshow(prof[:w].T, aspect="auto", origin="lower", cmap="RdBu",
               extent=[0, w*dt, 0, 1], vmin=-0.12, vmax=0.12)
ax.set_title("(a) FISH bend(station, time) — diagonal stripes = head→tail traveling wave")
ax.set_xlabel("time (s)"); ax.set_ylabel("body station  (0=head, 1=tail)")
plt.colorbar(im, ax=ax, label="lateral offset (BL)")

# (b) reference space-time heatmap (same colormap/scale)
ax = fig.add_subplot(2, 2, 2)
wr = min(len(ref), 180)
im = ax.imshow(ref[:wr].T, aspect="auto", origin="lower", cmap="RdBu",
               extent=[0, wr*dt, 0, 1], vmin=-0.12, vmax=0.12)
ax.set_title("(b) ZeF05 REFERENCE bend(station, time) — the imitation target")
ax.set_xlabel("time (s)"); ax.set_ylabel("body station"); plt.colorbar(im, ax=ax, label="lateral offset (BL)")

# (c) instantaneous profile snapshots (the body shape at a few times) - fish
ax = fig.add_subplot(2, 2, 3)
idxs = np.linspace(w//4, w//4 + int(1.0/dt), 6).astype(int)
idxs = idxs[idxs < T]
for i, t in enumerate(idxs):
    ax.plot(s, prof[t], "-o", ms=3, color=plt.cm.viridis(i/max(1, len(idxs)-1)), lw=1.5,
            label=f"t={t*dt:.2f}s" if i in (0, len(idxs)-1) else None)
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_title("(c) FISH instantaneous bend profiles over ~1 s (the body midline)")
ax.set_xlabel("body station (0=head → 1=tail)"); ax.set_ylabel("lateral offset (BL)")
ax.legend(fontsize=8)

# (d) amplitude envelope: fish (rms over time per station) vs reference envelope
ax = fig.add_subplot(2, 2, 4)
fish_env = prof.std(0) * np.sqrt(2)       # sqrt(2)*std ~ oscillation amplitude per station
ref_env_rms = ref.std(0) * np.sqrt(2)
ax.plot(s, fish_env, "b-o", ms=3, lw=1.8, label="FISH (ep_100) amplitude")
ax.plot(s, ref_env_rms, "r-s", ms=3, lw=1.8, label="ZeF05 reference amplitude")
ax.plot(s, env, "r:", lw=1.2, alpha=0.7, label="ref envelope A(s) [stored]")
ax.set_title("(d) bend AMPLITUDE envelope: fish vs reference (per station)")
ax.set_xlabel("body station (0=head → 1=tail)"); ax.set_ylabel("amplitude (BL)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
plt.savefig(args.out, dpi=110)
print(f"[bend-viz] saved {args.out}")
print(f"[bend-viz] fish tail-station amplitude {fish_env[-1]:.3f} BL vs reference {ref_env_rms[-1]:.3f} BL "
      f"(stored envelope tail {env[-1]:.3f})")
print(f"[bend-viz] fish head amp {fish_env[0]:.4f} tail amp {fish_env[-1]:.3f} -> tail/head ratio "
      f"{fish_env[-1]/max(fish_env[0],1e-6):.1f}x (reference {ref_env_rms[-1]/max(ref_env_rms[0],1e-6):.1f}x)")
