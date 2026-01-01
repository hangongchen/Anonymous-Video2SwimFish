#!/usr/bin/env python
"""AMP obs (Phi) sanity-check visualization -- sim rollout vs ZebraFish-05 reference, SAME code both.

Per sampled frame: LEFT = the fish in the world-horizontal, heading-aligned frame (the frame Phi is
computed in) with heading(red)/lateral(blue) axes; RIGHT = the Phi content -- the head->tail bend
profile as a CURVE (should visually match the fish's curve) + the proposed velocity channels
(fwd/lat/vert/yaw) as labeled bars, RAW and z-scored. Plus an overlay of sim-vs-ref ranges.

Checks: head->tail ordering, lateral sign, frame alignment, velocity signs, sim/ref scale match.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
sys.path.insert(0, "scripts")
from extract_amp_features import anterior_aligned_profile

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


K = 20
BL = 0.5
SEG = {0: (481, 599)}  # unused


def rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


# ---------------- SIM side: extract per-frame (heading-frame pts, bend profile, velocity channels) ----
def sim_frames(npz):
    d = np.load(npz)
    pc, root = d["pc"].astype(float), d["root"].astype(float)
    dt = float(d["step_dt"]) if "step_dt" in d.files else 1/30.
    out = []
    for t in range(len(pc)):
        rel = pc[t] - root[t, 0:3]
        fwd = rot(root[t, 3:7]) @ np.array([-1., 0, 0]); fwd[2] = 0
        fwd = fwd / (np.linalg.norm(fwd[:2]) + 1e-9)
        perp = np.array([-fwd[1], fwd[0], 0.])
        along = rel @ fwd; lateral = rel @ perp
        prof = anterior_aligned_profile(along, lateral, K)
        # velocity channels (world root vel -> heading frame)
        v = root[t, 7:10]
        vfwd = float(v @ fwd) / BL                 # forward (BL/s), +=head-first
        vlat = float(v @ perp) / BL                # lateral (BL/s)
        vvert = float(v[2]) / BL                   # vertical (BL/s)
        yaw = float(root[t, 12]) if root.shape[1] > 12 else 0.0   # world ang-vel z ~ yaw-rate
        out.append(dict(along=along, lateral=lateral, prof=prof,
                        vel=np.array([vfwd, vlat, vvert, yaw])))
    return out


# ---------------- REF side: bend from feat, pose from top mask, velocity from gt ----
def ref_frames():
    d = np.load("demo_out/zef05_amp_ref/amp_reference.npz")
    feat = d["feat"]; frames = d["frame"]; sp = d["speed_bl"]; yr = d["yaw_rate"]
    MROOT = "demo_out/zef05_seg/ZebraFish-05/imgT/mask"
    files = {int(os.path.basename(f)[:-4]): f for f in glob.glob(MROOT + "/*.png")}
    gt = np.loadtxt(_P("${ZEF_ROOT}/gt/gt.txt"), delimiter=",")
    gt = gt[gt[:, 1] == 1]
    headpx = {int(r[0]): np.array([r[5], r[6]]) for r in gt}
    pos3d = {int(r[0]): r[2:5] for r in gt}
    out = []
    for i, fr in enumerate(frames):
        fr = int(fr)
        # pose: top mask projected to heading frame (head->centroid axis, like the extractor)
        along = lateral = None
        if fr in files and fr in headpx:
            m = imageio.imread(files[fr]); m = m[..., 0] if m.ndim == 3 else m
            ys, xs = np.nonzero(m > 127)
            if len(xs) > 30:
                P = np.stack([xs, ys], 1).astype(float); head = headpx[fr]; ctr = P.mean(0)
                ax = ctr - head; ax /= np.linalg.norm(ax) + 1e-9; lat = np.array([-ax[1], ax[0]])
                rel = P - head; along = rel @ ax; lateral = rel @ lat
        # velocity: forward=speed_bl (mostly forward), lateral~0, vertical from 3d z, yaw=yaw_rate
        vvert = 0.0
        if fr in pos3d and int(frames[max(i-1,0)]) in pos3d:
            dz = pos3d[fr][2] - pos3d[int(frames[max(i-1,0)])][2]
            vvert = dz / 3.4 / (2/60.)             # BL/s (cm, 30fps stride-2)
        out.append(dict(along=along, lateral=lateral, prof=feat[i][:K],   # bend part of the 22-dim feat
                        vel=np.array([sp[i], 0.0, vvert, yr[i]]), fr=fr))
    return out


def pick(frames, key):
    """indices for straight, strong-left, strong-right, high-yaw."""
    tail = np.array([f["prof"][-1] for f in frames])   # tail lateral offset (bend sign)
    yaw = np.array([abs(f["vel"][3]) for f in frames])
    idx = {}
    idx["straight/low-bend"] = int(np.argmin(np.abs(tail) + 0.3*yaw))
    idx["strong LEFT bend"] = int(np.argmax(tail))
    idx["strong RIGHT bend"] = int(np.argmin(tail))
    idx["high YAW (turn)"] = int(np.argmax(yaw))
    return idx


def zscore(prof, mean, std):
    return (prof - mean) / std


def render(side, frames, mean, std, vmean, vstd, out):
    idx = pick(frames, side)
    labels = list(idx.keys())
    fig, ax = plt.subplots(len(labels), 3, figsize=(13, 3.1*len(labels)))
    vchan = ["fwd", "lat", "vert", "yaw"]
    for r, lab in enumerate(labels):
        f = frames[idx[lab]]
        aL, aM, aR = ax[r]
        # LEFT: pose in heading frame
        if f["along"] is not None:
            c = f["along"]
            aL.scatter(f["along"], f["lateral"], c=c, cmap="autumn", s=6)
        aL.annotate("", xy=(np.nanmax(f["along"])*0.9 if f["along"] is not None else 1, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        aL.text(0.05, 0.92, "heading (head→tail)", color="red", transform=aL.transAxes, fontsize=7)
        aL.annotate("", xy=(0, (np.nanmax(np.abs(f["lateral"]))*0.6 if f["along"] is not None else 1)),
                    xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        aL.text(0.05, 0.05, "lateral (+left)", color="blue", transform=aL.transAxes, fontsize=7)
        aL.set_aspect("equal"); aL.set_title(f"{side}: {lab}", fontsize=9); aL.set_xticks([]); aL.set_yticks([])
        # MID: bend profile curve raw + zscored
        s = np.linspace(0, 1, K)
        aM.plot(s, f["prof"], "C3-o", ms=3, label="raw bend (BL)")
        aM.plot(s, zscore(f["prof"], mean, std), "C0--", ms=2, label="z-scored", alpha=.6)
        aM.axhline(0, color="0.8", lw=1); aM.set_xlabel("head → tail (s)", fontsize=8)
        aM.set_ylabel("lateral offset", fontsize=8); aM.legend(fontsize=6); aM.set_title("Φ bend profile", fontsize=9)
        # RIGHT: velocity channels bars raw + zscored
        x = np.arange(4)
        aR.bar(x-0.2, f["vel"], 0.38, label="raw", color="C2")
        aR.bar(x+0.2, (f["vel"]-vmean)/vstd, 0.38, label="z-scored", color="C1", alpha=.7)
        for i, v in enumerate(f["vel"]):
            aR.text(i-0.2, v, f"{v:.2f}", ha="center", fontsize=6)
        aR.set_xticks(x); aR.set_xticklabels(vchan, fontsize=8); aR.axhline(0, color="0.8", lw=1)
        aR.legend(fontsize=6); aR.set_title("Φ velocity channels (proposed)", fontsize=9)
    plt.suptitle(f"AMP obs Φ -- {side}  (bend curve should match the fish curve on the left)", fontsize=12)
    plt.tight_layout(); plt.savefig(out, dpi=115); plt.close()
    print("wrote", out)
    return {lab: frames[idx[lab]] for lab in labels}


sim = sim_frames("demo_out/autoskel_amp/openwater_rollout.npz")
ref = ref_frames()
# shared normalization = REFERENCE stats (as in _init_disc)
ref_prof = np.stack([f["prof"] for f in ref])
mean, std = ref_prof.mean(0), ref_prof.std(0).clip(1e-4)
ref_vel = np.stack([f["vel"] for f in ref]); vmean, vstd = ref_vel.mean(0), ref_vel.std(0).clip(1e-4)

render("REFERENCE (ZebraFish-05)", ref, mean, std, vmean, vstd, "demo_out/zef05_amp_ref/amp_obs_viz_ref.png")
render("SIM (active rollout)", sim, mean, std, vmean, vstd, "demo_out/zef05_amp_ref/amp_obs_viz_sim.png")

# overlay: sim vs ref ranges
sim_prof = np.stack([f["prof"] for f in sim]); sim_vel = np.stack([f["vel"] for f in sim])
s = np.linspace(0, 1, K)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].fill_between(s, np.percentile(ref_prof,10,0), np.percentile(ref_prof,90,0), alpha=.3, color="k", label="reference 10-90%")
ax[0].fill_between(s, np.percentile(sim_prof,10,0), np.percentile(sim_prof,90,0), alpha=.3, color="C2", label="sim 10-90%")
ax[0].plot(s, ref_prof.mean(0), "k-", lw=2); ax[0].plot(s, sim_prof.mean(0), "C2-", lw=2)
ax[0].set_xlabel("head → tail"); ax[0].set_ylabel("bend (BL)"); ax[0].legend(); ax[0].set_title("Bend-profile range: sim vs reference")
vchan = ["fwd", "lat", "vert", "yaw"]
xx = np.arange(4)
ax[1].bar(xx-0.2, ref_vel.std(0), 0.38, label="reference std", color="k", alpha=.6)
ax[1].bar(xx+0.2, sim_vel.std(0), 0.38, label="sim std", color="C2", alpha=.7)
for i in range(4):
    ax[1].text(i-0.2, ref_vel.std(0)[i], f"μ{ref_vel.mean(0)[i]:.2f}", ha="center", fontsize=7)
    ax[1].text(i+0.2, sim_vel.std(0)[i], f"μ{sim_vel.mean(0)[i]:.2f}", ha="center", fontsize=7)
ax[1].set_xticks(xx); ax[1].set_xticklabels(vchan); ax[1].legend(); ax[1].set_title("Velocity-channel spread (std) + mean: sim vs reference")
plt.suptitle("Sim vs Reference obs ranges -- a big mismatch lets D win on an artifact, not motion", fontsize=12)
plt.tight_layout(); plt.savefig("demo_out/zef05_amp_ref/amp_obs_viz_overlay.png", dpi=115); plt.close()
print("wrote demo_out/zef05_amp_ref/amp_obs_viz_overlay.png")

# print numeric ranges for the report
print("\n=== channel ranges (raw) ===")
print("bend tail p10/p90: ref %.3f/%.3f | sim %.3f/%.3f"%(
    np.percentile(ref_prof[:,-1],10),np.percentile(ref_prof[:,-1],90),
    np.percentile(sim_prof[:,-1],10),np.percentile(sim_prof[:,-1],90)))
for i,ch in enumerate(vchan):
    print("  %-4s ref mean %.3f std %.3f | sim mean %.3f std %.3f"%(
        ch, ref_vel.mean(0)[i], ref_vel.std(0)[i], sim_vel.mean(0)[i], sim_vel.std(0)[i]))
