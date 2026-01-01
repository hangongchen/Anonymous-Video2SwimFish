#!/usr/bin/env python
"""How is TURNING represented in the zebrafish curvature manifold?

Works ONLY from the frozen files:
  demo_out/zef_manifold/curvature_dataset.npz
  demo_out/zef_manifold/pca_basis.npz          (canonical PCA, never refit)
  ~/hangong/3D-ZeF/data/ZebraFish-05/gt/gt.txt (world-cm head track)

Outputs -> demo_out/zef_manifold/turning/
"""
import json
import os
import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


# ---------------------------------------------------------------- paths / consts
ROOT = _P("${FISH_ROOT}/demo_out/zef_manifold")
OUT = os.path.join(ROOT, "turning")
os.makedirs(OUT, exist_ok=True)
GT = _P("${ZEF_ROOT}/gt/gt.txt")

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
DPI = 140
FPS = 60.0
DT = 1.0 / FPS
SG_WIN = 9          # ~0.15 s at 60 fps (odd)
SG_ORD = 2

metrics = {}

# ---------------------------------------------------------------- load frozen data
d = np.load(os.path.join(ROOT, "curvature_dataset.npz"))
kappa = d["kappa_bl"].astype(np.float64)        # (900,20) kappa*BL
theta = d["theta"].astype(np.float64)           # (900,20)
valid = d["valid"].astype(bool)                 # 897 True
t = d["t_sec"].astype(np.float64)               # (900,)
N, NS = kappa.shape

p = np.load(os.path.join(ROOT, "pca_basis.npz"))
mean_k = p["mean"].astype(np.float64)           # (20,)
comps = p["components"].astype(np.float64)      # (20,20) rows = PCs
coeffs = p["coeffs"].astype(np.float64)         # (900,20), NaN on invalid


def fill_gaps(x):
    """Linear-interpolate NaNs over time (the 3 invalid frames)."""
    x = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        idx = np.arange(len(x))
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    return x


# ================================================================ 1. heading & yaw rate
# --- velocity heading from GT world-cm horizontal head track
gt = np.loadtxt(GT, delimiter=",")
order = np.argsort(gt[:, 0])
gt = gt[order]
assert gt.shape[0] == N
xw = savgol_filter(gt[:, 2], SG_WIN, SG_ORD)    # 3d_x cm, smoothed ~0.15 s
yw = savgol_filter(gt[:, 3], SG_WIN, SG_ORD)    # 3d_y cm
vx = np.gradient(xw) * FPS
vy = np.gradient(yw) * FPS
speed = np.hypot(vx, vy)                        # cm/s
head_vel = np.unwrap(np.arctan2(vy, vx))        # rad, unwrapped over time
head_vel_s = savgol_filter(head_vel, SG_WIN, SG_ORD)
yaw_vel = savgol_filter(head_vel, SG_WIN, SG_ORD, deriv=1, delta=DT)  # rad/s

# --- body heading from theta[:, 0:3].mean(1); per-frame only -> unwrap over TIME
h_raw = np.nanmean(theta[:, 0:3], axis=1)       # NaN on the 3 invalid frames
h_fill = fill_gaps(h_raw)                       # linear interp over 3 NaN frames
head_body = np.unwrap(h_fill)                   # continuous across time
head_body_s = savgol_filter(head_body, SG_WIN, SG_ORD)
yaw_body = savgol_filter(head_body, SG_WIN, SG_ORD, deriv=1, delta=DT)  # rad/s


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


r_yaw_all = pearson(yaw_body, yaw_vel)
moving = speed > 2.0                            # velocity heading unreliable when coasting
r_yaw_moving = pearson(yaw_body[moving], yaw_vel[moving])
metrics["yaw_corr_body_vs_vel_raw"] = r_yaw_all
metrics["yaw_corr_body_vs_vel_raw_moving_gt2cms"] = r_yaw_moving
metrics["moving_frac_speed_gt2cms"] = float(moving.mean())

# The image frame is mirrored w.r.t. the world x-y frame (r < 0). Keep the BODY
# signals untouched (they share the image convention with kappa); mirror-correct
# the VELOCITY heading for comparison only.
mirror = r_yaw_all < 0
SGN = -1.0 if mirror else 1.0
yaw_vel_al = SGN * yaw_vel
head_vel_al = SGN * head_vel_s
metrics["world_image_mirror_detected"] = bool(mirror)
metrics["yaw_corr_after_mirror_all"] = pearson(yaw_body, yaw_vel_al)
metrics["yaw_corr_after_mirror_moving"] = pearson(yaw_body[moving], yaw_vel_al[moving])
yaw_body_al = yaw_body                          # primary signal, image convention
head_body_al = head_body_s

# --- plot: headings + yaw rates
fig, ax = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
hv = head_vel_al - head_vel_al[0]
hb = head_body_al - head_body_al[0]
ax[0].plot(t, hv, color=OI[0], lw=1.4, label="velocity heading (GT track, mirror-corrected)")
ax[0].plot(t, hb, color=OI[1], lw=1.4, label="body heading (theta[:,0:3], image frame)")
ax[0].set_ylabel("heading, unwrapped (rad)")
ax[0].legend(loc="best", fontsize=9)
ax[0].set_title("Heading and yaw rate: GT-velocity vs body-tangent estimate")
ax[1].plot(t, yaw_vel_al, color=OI[0], lw=1.0, label="yaw rate from velocity heading (mirror-corrected)")
ax[1].plot(t, yaw_body_al, color=OI[1], lw=1.2, label="yaw rate from body heading")
ax[1].axhline(0, color="0.6", lw=0.6)
ax[1].set_ylabel("yaw rate (rad/s)")
ax[1].set_xlabel("time (s)")
ax[1].legend(loc="best", fontsize=9)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "heading_yaw.png"), dpi=DPI)
plt.close(fig)

# ================================================================ 2. turning frames
turn1 = np.abs(yaw_body_al) > 1.0
turn2 = np.abs(yaw_body_al) > 2.0
metrics["turning_frac_1.0rad_s"] = float(turn1.mean())
metrics["turning_frac_2.0rad_s"] = float(turn2.mean())
metrics["yaw_body_abs_max_rad_s"] = float(np.abs(yaw_body_al).max())
metrics["yaw_body_abs_median_rad_s"] = float(np.median(np.abs(yaw_body_al)))


def episodes(mask, merge_gap=3, min_len=3):
    """Contiguous True runs; merge gaps <= merge_gap frames, drop runs < min_len."""
    m = mask.copy().astype(int)
    idx = np.flatnonzero(np.diff(np.r_[0, m, 0]))
    runs = [(idx[i], idx[i + 1]) for i in range(0, len(idx), 2)]
    merged = []
    for s0, e0 in runs:
        if merged and s0 - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e0)
        else:
            merged.append((s0, e0))
    return [(s0, e0) for s0, e0 in merged if e0 - s0 >= min_len]


eps1 = episodes(turn1)
metrics["turning_episodes_1.0rad_s"] = len(eps1)
metrics["turning_episode_mean_dur_s"] = float(np.mean([(e - s) * DT for s, e in eps1]))
eps2 = episodes(turn2)
metrics["turning_episodes_2.0rad_s"] = len(eps2)

# ================================================================ 3. DC bend vs yaw rate
dc_bend = fill_gaps(np.nanmean(kappa, axis=1))  # mean over s of kappa*BL per frame

max_lag = 60  # +-1 s
lags = np.arange(-max_lag, max_lag + 1)


def xcorr(a, b):
    """r(L) = corr(a(t), b(t+L)); positive L -> a LEADS b."""
    out = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L >= 0:
            out[i] = pearson(a[: N - L], b[L:])
        else:
            out[i] = pearson(a[-L:], b[: N + L])
    return out


xc = xcorr(dc_bend, yaw_body_al)
ipk = int(np.argmax(np.abs(xc)))
metrics["xcorr_peak_r"] = float(xc[ipk])
metrics["xcorr_peak_lag_frames"] = int(lags[ipk])
metrics["xcorr_peak_lag_ms"] = float(lags[ipk] * DT * 1000.0)
metrics["xcorr_zero_lag_r"] = float(xc[max_lag])

# per-episode bias: mean a1 / mean DC bend vs mean yaw rate over each turning episode
a1_fill = fill_gaps(coeffs[:, 0])
ep_yaw = np.array([yaw_body_al[s0:e0].mean() for s0, e0 in eps1])
ep_a1 = np.array([a1_fill[s0:e0].mean() for s0, e0 in eps1])
ep_dc = np.array([dc_bend[s0:e0].mean() for s0, e0 in eps1])
metrics["episode_r_meanA1_vs_meanYaw"] = pearson(ep_a1, ep_yaw)
metrics["episode_r_meanDCbend_vs_meanYaw"] = pearson(ep_dc, ep_yaw)

fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))
ax[0].plot(lags * DT * 1000.0, xc, color=OI[0], lw=1.6)
ax[0].axvline(0, color="0.6", lw=0.7)
ax[0].axhline(0, color="0.6", lw=0.7)
ax[0].plot(lags[ipk] * DT * 1000.0, xc[ipk], "o", color=OI[1], ms=7,
           label=f"peak r={xc[ipk]:+.3f} @ {lags[ipk]*DT*1000:+.0f} ms")
ax[0].set_xlabel("lag of yaw rate behind DC bend (ms)  [positive = bend leads]")
ax[0].set_ylabel("Pearson r")
ax[0].set_title("Cross-correlation: DC bend vs body yaw rate")
ax[0].legend(loc="best", fontsize=9)
ax[1].scatter(ep_yaw, ep_a1, s=42, color=OI[1], edgecolor="k", lw=0.5,
              label=f"episodes (n={len(eps1)}), r={metrics['episode_r_meanA1_vs_meanYaw']:+.3f}")
ax[1].axhline(0, color="0.6", lw=0.7)
ax[1].axvline(0, color="0.6", lw=0.7)
ax[1].set_xlabel("episode-mean yaw rate (rad/s)")
ax[1].set_ylabel("episode-mean $a_1$ (kappa*BL units)")
ax[1].set_title("Per-turning-episode bias of $a_1$")
ax[1].legend(loc="best", fontsize=9)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dcbend_yaw_xcorr.png"), dpi=DPI)
plt.close(fig)

# DC bend is essentially which PC?
a_valid = coeffs[valid]
r_dc_a1 = pearson(dc_bend[valid], coeffs[valid, 0])
r_dc_a2 = pearson(dc_bend[valid], coeffs[valid, 1])
metrics["r_dcbend_a1"] = r_dc_a1
metrics["r_dcbend_a2"] = r_dc_a2

# ================================================================ 4. which PCs carry turning?
turn_v = turn1 & valid
straight_v = (~turn1) & valid
metrics["n_turning_valid"] = int(turn_v.sum())
metrics["n_straight_valid"] = int(straight_v.sum())

# (a) distributions of a1..a4, straight vs turning -------------------------------
fig, ax = plt.subplots(figsize=(9.2, 4.8))
pos_s = np.arange(4) * 3.0
pos_t = pos_s + 1.0
vp_s = ax.violinplot([coeffs[straight_v, j] for j in range(4)], positions=pos_s,
                     widths=0.9, showextrema=False, showmedians=True)
vp_t = ax.violinplot([coeffs[turn_v, j] for j in range(4)], positions=pos_t,
                     widths=0.9, showextrema=False, showmedians=True)
for b in vp_s["bodies"]:
    b.set_facecolor(OI[0]); b.set_alpha(0.75)
vp_s["cmedians"].set_color("k")
for b in vp_t["bodies"]:
    b.set_facecolor(OI[1]); b.set_alpha(0.75)
vp_t["cmedians"].set_color("k")
ax.axhline(0, color="0.6", lw=0.7)
ax.set_xticks(pos_s + 0.5)
ax.set_xticklabels([f"$a_{j+1}$" for j in range(4)])
ax.set_ylabel("PC coefficient (kappa*BL units)")
ax.set_title("PC coefficients: straight vs turning frames (|yaw| > 1 rad/s)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=OI[0], alpha=0.75, label=f"straight (n={straight_v.sum()})"),
                   Patch(color=OI[1], alpha=0.75, label=f"turning (n={turn_v.sum()})")],
          loc="best", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pc_dist_straight_vs_turning.png"), dpi=DPI)
plt.close(fig)

for j in range(4):
    metrics[f"a{j+1}_std_straight"] = float(coeffs[straight_v, j].std())
    metrics[f"a{j+1}_std_turning"] = float(coeffs[turn_v, j].std())
    metrics[f"a{j+1}_absmean_turning"] = float(np.abs(coeffs[turn_v, j]).mean())

# (b) variance/energy per PC within turning frames (canonical basis projection) --
def energy_frac(mask):
    e = np.mean(coeffs[mask] ** 2, axis=0)      # 2nd moment about the canonical mean
    return e / e.sum()

ef_turn = energy_frac(turn_v)
ef_straight = energy_frac(straight_v)
for j in range(6):
    metrics[f"turning_energy_frac_PC{j+1}"] = float(ef_turn[j])
metrics["turning_energy_frac_PC1+2"] = float(ef_turn[:2].sum())
metrics["turning_energy_frac_PC1..4"] = float(ef_turn[:4].sum())
metrics["straight_energy_frac_PC1+2"] = float(ef_straight[:2].sum())

fig, ax = plt.subplots(figsize=(8.4, 4.4))
jj = np.arange(1, 9)
w = 0.38
ax.bar(jj - w / 2, ef_straight[:8], width=w, color=OI[0], label="straight frames")
ax.bar(jj + w / 2, ef_turn[:8], width=w, color=OI[1], label="turning frames")
ax.set_yscale("log")
ax.set_xlabel("PC index")
ax.set_ylabel("fraction of curvature energy (log)")
ax.set_title("Energy per canonical PC, straight vs turning (|yaw| > 1 rad/s)")
ax.legend(loc="best", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pc_energy_turn_vs_straight.png"), dpi=DPI)
plt.close(fig)

# (c) low-pass (<1 Hz) vs high-pass (>1.5 Hz) split of a1, a2 (and a3, a4) -------
b_lp, a_lp = butter(3, 1.0 / (FPS / 2), "low")
b_hp, a_hp = butter(3, 1.5 / (FPS / 2), "high")
a_f = {}
for j in range(4):
    aj = fill_gaps(coeffs[:, j])
    a_f[j] = dict(raw=aj, lp=filtfilt(b_lp, a_lp, aj), hp=filtfilt(b_hp, a_hp, aj))
    P_lp_t = float(np.mean(a_f[j]["lp"][turn1] ** 2))
    P_lp_s = float(np.mean(a_f[j]["lp"][~turn1] ** 2))
    P_hp_t = float(np.mean(a_f[j]["hp"][turn1] ** 2))
    P_hp_s = float(np.mean(a_f[j]["hp"][~turn1] ** 2))
    metrics[f"a{j+1}_LPpower_turning"] = P_lp_t
    metrics[f"a{j+1}_LPpower_straight"] = P_lp_s
    metrics[f"a{j+1}_LPpower_ratio_turn_over_straight"] = P_lp_t / P_lp_s
    metrics[f"a{j+1}_HPpower_turning"] = P_hp_t
    metrics[f"a{j+1}_HPpower_straight"] = P_hp_s
    metrics[f"a{j+1}_HPpower_ratio_turn_over_straight"] = P_hp_t / P_hp_s
metrics["r_a1lp_yaw"] = pearson(a_f[0]["lp"], yaw_body_al)
metrics["r_a2lp_yaw"] = pearson(a_f[1]["lp"], yaw_body_al)
xc_a1 = xcorr(a_f[0]["raw"], yaw_body_al)
i1 = int(np.argmax(np.abs(xc_a1)))
metrics["xcorr_a1raw_yaw_peak_r"] = float(xc_a1[i1])
metrics["xcorr_a1raw_yaw_peak_lag_ms"] = float(lags[i1] * DT * 1000.0)
xc_a1lp = xcorr(a_f[0]["lp"], yaw_body_al)
i2 = int(np.argmax(np.abs(xc_a1lp)))
metrics["xcorr_a1lp_yaw_peak_r"] = float(xc_a1lp[i2])
metrics["xcorr_a1lp_yaw_peak_lag_ms"] = float(lags[i2] * DT * 1000.0)

fig, ax = plt.subplots(figsize=(8.4, 4.4))
jj = np.arange(1, 5)
w = 0.2
ax.bar(jj - 1.5 * w, [metrics[f"a{j}_LPpower_straight"] for j in jj], width=w,
       color=OI[0], label="LP <1 Hz, straight")
ax.bar(jj - 0.5 * w, [metrics[f"a{j}_LPpower_turning"] for j in jj], width=w,
       color=OI[1], label="LP <1 Hz, turning")
ax.bar(jj + 0.5 * w, [metrics[f"a{j}_HPpower_straight"] for j in jj], width=w,
       color=OI[5], label="HP >1.5 Hz, straight")
ax.bar(jj + 1.5 * w, [metrics[f"a{j}_HPpower_turning"] for j in jj], width=w,
       color=OI[4], label="HP >1.5 Hz, turning")
ax.set_yscale("log")
ax.set_xticks(jj)
ax.set_xticklabels([f"$a_{j}$" for j in jj])
ax.set_ylabel("mean power ((kappa*BL)^2, log)")
ax.set_title("Band-split PC-coefficient power: turning vs straight")
ax.legend(loc="best", fontsize=8, ncol=2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pc_band_power.png"), dpi=DPI)
plt.close(fig)

# (d) reconstruction error, 2 vs 4 PCs, on turning frames ------------------------
def recon_rms(mask, n):
    rec = mean_k + coeffs[mask][:, :n] @ comps[:n, :]
    return float(np.sqrt(np.mean((kappa[mask] - rec) ** 2)))

rms_k_turn = float(np.sqrt(np.mean(kappa[turn_v] ** 2)))
rms_k_straight = float(np.sqrt(np.mean(kappa[straight_v] ** 2)))
for n in (2, 4):
    metrics[f"recon_rms_turning_{n}PC"] = recon_rms(turn_v, n)
    metrics[f"recon_rms_straight_{n}PC"] = recon_rms(straight_v, n)
metrics["rms_kappa_turning"] = rms_k_turn
metrics["rms_kappa_straight"] = rms_k_straight
metrics["recon_relerr_turning_2PC"] = metrics["recon_rms_turning_2PC"] / rms_k_turn
metrics["recon_relerr_turning_4PC"] = metrics["recon_rms_turning_4PC"] / rms_k_turn
metrics["recon_relerr_straight_2PC"] = metrics["recon_rms_straight_2PC"] / rms_k_straight

# ================================================================ 5. time-locked overlay
fig, ax = plt.subplots(3, 1, figsize=(12.5, 8.2), sharex=True)
for a in ax:
    for s0, e0 in eps1:
        a.axvspan(t[s0], t[min(e0, N - 1)], color=OI[3], alpha=0.22, lw=0)
ax[0].plot(t, yaw_body_al, color=OI[0], lw=1.2, label="body yaw rate")
ax[0].axhline(1.0, color="0.5", lw=0.7, ls="--")
ax[0].axhline(-1.0, color="0.5", lw=0.7, ls="--", label="±1 rad/s threshold")
ax[0].set_ylabel("yaw rate (rad/s)")
ax[0].legend(loc="upper right", fontsize=9)
ax[0].set_title("Turning, DC bend, and the slow component of $a_1$ "
                "(pink spans = turning episodes, |yaw| > 1 rad/s)")
ax[1].plot(t, dc_bend, color=OI[2], lw=1.2, label="DC bend  $\\langle\\kappa\\,BL\\rangle_s$")
ax[1].axhline(0, color="0.6", lw=0.6)
ax[1].set_ylabel("mean kappa*BL (-)")
ax[1].legend(loc="upper right", fontsize=9)
ax[2].plot(t, a_f[0]["raw"], color="0.75", lw=0.8, label="$a_1$ raw")
ax[2].plot(t, a_f[0]["lp"], color=OI[1], lw=1.6, label="$a_1$ low-pass < 1 Hz")
ax[2].axhline(0, color="0.6", lw=0.6)
ax[2].set_ylabel("$a_1$ (kappa*BL units)")
ax[2].set_xlabel("time (s)")
ax[2].legend(loc="upper right", fontsize=9)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "timelocked_yaw_bend_a1.png"), dpi=DPI)
plt.close(fig)

# ================================================================ save numbers
with open(os.path.join(OUT, "turning_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

for k, v in metrics.items():
    print(f"{k:45s} {v}")
