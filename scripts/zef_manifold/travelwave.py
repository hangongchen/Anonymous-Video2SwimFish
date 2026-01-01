#!/usr/bin/env python
"""Traveling-wave model fit + residual analysis for the ZebraFish-05 curvature dataset.

Model:  kappa_fit(s,t) = alpha_b * A(s) * g(omega_b*(t-t_b0) - k*s + phi0_b)  inside burst b
        kappa_fit = 0 outside bursts (the model has NO coast / turn story).

A(s): global amplitude envelope (complex demodulation, aggregated over bursts).
k   : global wavenumber (rad/BL) from weighted linear fit of the demodulated phase phi(s).
g   : global 2*pi-periodic waveform from phase-folding the wide-band tail signal on the
      narrow-band Hilbert phase (folding the NARROW-band signal on its own Hilbert phase
      would force g -> cos by construction, so we fold the wide-band signal instead).
Per burst: omega_b (dominant frequency), phi0_b + alpha_b by least squares (grid on phi0).

Outputs -> ${FISH_ROOT}/demo_out/zef_manifold/travelwave/
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


# ----------------------------------------------------------------------------- config
DATA = _P("${FISH_ROOT}/demo_out/zef_manifold/curvature_dataset.npz")
PCA = _P("${FISH_ROOT}/demo_out/zef_manifold/pca_basis.npz")
OUT = _P("${FISH_ROOT}/demo_out/zef_manifold/travelwave")
os.makedirs(OUT, exist_ok=True)

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
GRID = dict(color="0.85", lw=0.6)
DPI = 150

BEAT_LO, BEAT_HI = 1.5, 5.0     # Hz, beat band (task: bursts ~2-4 Hz; margin both sides)
ENV_LP = 2.0                    # Hz, envelope smoothing low-pass
THR = 0.35                      # kappa*BL units, burst threshold on smoothed envelope
MIN_DUR = 0.25                  # s, minimum burst duration (~1 beat period at 4 Hz)
MERGE_GAP = 0.15                # s, merge bursts separated by less than this
NBINS = 48                      # phase bins for g
PHI0_GRID = 256                 # grid resolution for per-burst phi0 fit

results = {}

# ----------------------------------------------------------------------------- load
d = np.load(DATA)
kappa = d["kappa_bl"].astype(np.float64)      # (900, 20)
valid = d["valid"].astype(bool)
t = d["t_sec"].astype(np.float64)
fps = float(d["fps"])
s = d["s_stations"].astype(np.float64)
N, S = kappa.shape
dt = 1.0 / fps

# gap-fill the 3 invalid frames by per-station linear interpolation in time
# (needed for filtering / Hilbert; all reported statistics EXCLUDE these frames)
kfill = kappa.copy()
for j in range(S):
    col = kfill[:, j]
    bad = ~np.isfinite(col)
    col[bad] = np.interp(t[bad], t[~bad], col[~bad])
results["n_invalid_interpolated"] = int((~valid).sum())

# ----------------------------------------------------------------------------- filters
sos_bp = signal.butter(3, [BEAT_LO, BEAT_HI], btype="bandpass", fs=fps, output="sos")
sos_env = signal.butter(2, ENV_LP, btype="low", fs=fps, output="sos")
sos_dc = signal.butter(3, 1.0, btype="low", fs=fps, output="sos")     # turn / DC trend
sos_hp = signal.butter(3, 1.0, btype="high", fs=fps, output="sos")    # wide-band (no DC)

tail = kfill[:, -3:].mean(axis=1)             # tail-region curvature (last 3 stations)
tail_bp = signal.sosfiltfilt(sos_bp, tail)
analytic = signal.hilbert(tail_bp)
env = np.abs(analytic)
env_s = np.maximum(signal.sosfiltfilt(sos_env, env), 0.0)
inst_phase = np.unwrap(np.angle(analytic))
inst_freq = np.gradient(inst_phase, dt) / (2 * np.pi)

kappa_bp = signal.sosfiltfilt(sos_bp, kfill, axis=0)   # per-station beat-band signal
tail_wb = signal.sosfiltfilt(sos_hp, tail)             # wide-band tail (keeps harmonics)
dc_bend = kfill.mean(axis=1)                            # DC bend (mean kappa over s)
yaw_proxy = signal.sosfiltfilt(sos_dc, dc_bend)         # low-passed DC bend

# ----------------------------------------------------------------------------- 1. bursts
on = env_s > THR
# merge short gaps
edges = np.flatnonzero(np.diff(on.astype(int)))
runs = []
i0 = 0 if on[0] else None
starts = list(np.flatnonzero(np.diff(on.astype(int)) == 1) + 1)
stops = list(np.flatnonzero(np.diff(on.astype(int)) == -1) + 1)
if on[0]:
    starts = [0] + starts
if on[-1]:
    stops = stops + [N]
spans = list(zip(starts, stops))
# merge gaps < MERGE_GAP
merged = []
for a, b in spans:
    if merged and (a - merged[-1][1]) * dt < MERGE_GAP:
        merged[-1] = (merged[-1][0], b)
    else:
        merged.append((a, b))
# min duration
bursts = [(a, b) for a, b in merged if (b - a) * dt >= MIN_DUR]
burst_mask = np.zeros(N, bool)
for a, b in bursts:
    burst_mask[a:b] = True

results["burst_threshold_kappaBL"] = THR
results["burst_min_duration_s"] = MIN_DUR
results["burst_merge_gap_s"] = MERGE_GAP
results["n_bursts"] = len(bursts)
results["burst_fraction_of_time"] = float(burst_mask.mean())
results["burst_durations_s"] = [round((b - a) * dt, 3) for a, b in bursts]

# ----------------------------------------------------------------------------- 2. per-burst frequency
burst_info = []
for a, b in bursts:
    seg = tail_bp[a:b]
    w = np.hanning(len(seg))
    F = np.fft.rfft(seg * w, n=4096)
    fr = np.fft.rfftfreq(4096, dt)
    sel = (fr >= BEAT_LO) & (fr <= BEAT_HI)
    fpk = fr[sel][np.argmax(np.abs(F[sel]))]
    if_med = float(np.median(inst_freq[a:b]))
    energy = float((env_s[a:b] ** 2).sum())
    burst_info.append(dict(a=int(a), b=int(b), f_dom=float(fpk), f_if=if_med,
                           energy=energy, dur=(b - a) * dt,
                           amp=float(env_s[a:b].mean())))

f_dom = np.array([bi["f_dom"] for bi in burst_info])
wts = np.array([bi["energy"] for bi in burst_info])
f_global = float(np.sum(f_dom * wts) / wts.sum())
omega_global = 2 * np.pi * f_global
results["per_burst_f_dom_Hz"] = [round(float(x), 3) for x in f_dom]
results["f_spread_Hz"] = dict(min=float(f_dom.min()), max=float(f_dom.max()),
                              mean=float(f_dom.mean()), std=float(f_dom.std()))
results["f_global_Hz"] = f_global
results["omega_global_rad_s"] = omega_global

# ----------------------------------------------------------------------------- 3. A(s), k by complex demodulation
Zb = np.zeros((len(bursts), S), complex)
for ib, bi in enumerate(burst_info):
    a, b = bi["a"], bi["b"]
    tt = t[a:b] - t[a]
    w = np.hanning(b - a)
    carrier = np.exp(-1j * 2 * np.pi * bi["f_dom"] * tt)
    Zb[ib] = 2.0 * (kappa_bp[a:b].T * (w * carrier)).sum(axis=1) / w.sum()

absZ = np.abs(Zb)
ref_station = int(np.argmax((absZ * wts[:, None]).sum(0)))   # highest-SNR station
Zal = Zb * np.exp(-1j * np.angle(Zb[:, ref_station]))[:, None]   # align tail phase to 0
Zmean = (Zal * wts[:, None]).sum(0) / wts.sum()

A_raw = (absZ * wts[:, None]).sum(0) / wts.sum()   # robust envelope (mean of |Z|)
A_env = A_raw / A_raw.max()
coherence = np.abs(Zmean) / A_raw                   # phase coherence across bursts, per station


def unwrap_from_tail(ph):
    return np.unwrap(ph[::-1])[::-1]


phi = unwrap_from_tail(np.angle(Zmean))
phi = phi - phi[ref_station]

# weighted linear fit  phi(s) = phi0 - k*s , weights = aggregated |Z|
wj = A_raw
Wm = np.diag(wj)
X = np.column_stack([np.ones(S), s])
beta, *_ = np.linalg.lstsq(Wm @ X, Wm @ phi, rcond=None)
phi0_fit, slope = beta
k_global = float(-slope)                     # rad / BL
phi_hat = X @ beta
ssr = np.sum(wj * (phi - phi_hat) ** 2)
sst = np.sum(wj * (phi - np.average(phi, weights=wj)) ** 2)
r2_phase = float(1 - ssr / sst)

# quadratic fit -> does the wave accelerate/decelerate along the body?
X2 = np.column_stack([np.ones(S), s, s ** 2])
b2, *_ = np.linalg.lstsq(Wm @ X2, Wm @ phi, rcond=None)
phi_q = X2 @ b2
r2_phase_quad = float(1 - np.sum(wj * (phi - phi_q) ** 2) / sst)
k_local_head = float(-(b2[1] + 2 * b2[2] * 0.25))   # local k at s=0.25
k_local_tail = float(-(b2[1] + 2 * b2[2] * 0.75))   # local k at s=0.75

# per-burst k spread
k_per_burst = []
for ib in range(len(bursts)):
    ph_b = unwrap_from_tail(np.angle(Zal[ib]))
    ph_b -= ph_b[ref_station]
    wjb = absZ[ib]
    bb, *_ = np.linalg.lstsq(np.diag(wjb) @ X, np.diag(wjb) @ ph_b, rcond=None)
    k_per_burst.append(float(-bb[1]))
k_per_burst = np.array(k_per_burst)

lam = 2 * np.pi / k_global
c_wave = omega_global / k_global
results["k_rad_per_BL"] = k_global
results["lambda_BL"] = lam
results["wave_speed_BL_s"] = c_wave
results["phase_R2_linear"] = r2_phase
results["phase_R2_quadratic"] = r2_phase_quad
results["k_local_s0.25"] = k_local_head
results["k_local_s0.75"] = k_local_tail
results["k_per_burst_mean"] = float(np.mean(k_per_burst))
results["k_per_burst_std"] = float(np.std(k_per_burst))
results["phase_coherence_tail_half"] = float(coherence[S // 2:].mean())
results["total_phase_drop_head_to_tail_rad"] = float(phi[0] - phi[-1])
results["wavelengths_visible_on_body"] = float((phi[0] - phi[-1]) / (2 * np.pi))

# ----------------------------------------------------------------------------- 4. waveform g
# fold the WIDE-band tail signal (harmonics preserved) on the narrow-band Hilbert
# phase; normalize by the smoothed envelope so every burst contributes equal weight
ph_wrap = np.mod(inst_phase, 2 * np.pi)
norm_sig = tail_wb / np.maximum(env_s, 1e-3)
bin_id = np.minimum((ph_wrap / (2 * np.pi) * NBINS).astype(int), NBINS - 1)
g_mean = np.zeros(NBINS)
g_std = np.zeros(NBINS)
for i in range(NBINS):
    m = burst_mask & (bin_id == i)
    g_mean[i] = norm_sig[m].mean()
    g_std[i] = norm_sig[m].std()
phase_c = (np.arange(NBINS) + 0.5) / NBINS * 2 * np.pi

G = np.fft.rfft(g_mean)
h1, h2, h3 = np.abs(G[1]), np.abs(G[2]), np.abs(G[3])
results["g_harmonic_power_2f_over_f"] = float((h2 / h1) ** 2)
results["g_harmonic_power_3f_over_f"] = float((h3 / h1) ** 2)
results["g_amp_ratio_2f_over_f"] = float(h2 / h1)
results["g_amp_ratio_3f_over_f"] = float(h3 / h1)

g_norm = g_mean / np.abs(g_mean).max()          # model waveform, peak = 1


def g_fun(phase):
    """periodic linear interpolation of the binned waveform"""
    pc = np.concatenate([phase_c - 2 * np.pi, phase_c, phase_c + 2 * np.pi])
    gv = np.tile(g_norm, 3)
    return np.interp(np.mod(phase, 2 * np.pi), pc, gv)


# ----------------------------------------------------------------------------- 5. model fit
kappa_fit = np.zeros_like(kfill)
phi0_grid = np.linspace(0, 2 * np.pi, PHI0_GRID, endpoint=False)
per_burst_fit = []
for ib, bi in enumerate(burst_info):
    a, b = bi["a"], bi["b"]
    tt = t[a:b] - t[a]
    y = kfill[a:b] - kfill[a:b].mean(axis=0)    # model has no DC: fit to demeaned kappa
    base = 2 * np.pi * bi["f_dom"] * tt[:, None] - k_global * s[None, :]
    best = (np.inf, 0.0, 0.0)
    for p0 in phi0_grid:
        F = A_env[None, :] * g_fun(base + p0)
        num = float((F * y).sum())
        den = float((F * F).sum())
        alpha = num / den if den > 0 else 0.0
        sse = float(((y - alpha * F) ** 2).sum())
        if sse < best[0]:
            best = (sse, p0, alpha)
    sse, p0, alpha = best
    F = A_env[None, :] * g_fun(base + p0)
    kappa_fit[a:b] = alpha * F
    sst_b = float((y ** 2).sum())
    per_burst_fit.append(dict(phi0=float(p0), alpha=float(alpha),
                              r2=float(1 - sse / sst_b), dur=bi["dur"], f=bi["f_dom"]))

results["alpha_per_burst"] = [round(pb["alpha"], 3) for pb in per_burst_fit]
results["r2_per_burst_demeaned"] = [round(pb["r2"], 3) for pb in per_burst_fit]

# variance explained (valid frames only; per-station temporal means as baseline,
# same convention as the canonical PCA)
def var_explained(mask):
    K = kappa[mask]                 # raw (with NaN excluded via mask upstream)
    Fh = kappa_fit[mask]
    mu = K.mean(axis=0)
    sstot = float(((K - mu) ** 2).sum())
    ssres = float(((K - Fh) ** 2).sum())
    return 1 - ssres / sstot, np.sqrt(ssres / K.size), np.sqrt(sstot / K.size)


mask_burst = burst_mask & valid
mask_all = valid
ve_b, rmse_b, rmsd_b = var_explained(mask_burst)
ve_a, rmse_a, rmsd_a = var_explained(mask_all)
results["VE_burst_frames"] = float(ve_b)
results["VE_all_valid_frames"] = float(ve_a)
results["RMSE_burst_frames"] = float(rmse_b)
results["RMSE_all_valid_frames"] = float(rmse_a)
results["RMS_data_burst_frames"] = float(rmsd_b)
results["RMS_data_all_valid"] = float(rmsd_a)

# ----------------------------------------------------------------------------- 5b. DIAGNOSTICS: where does the rigid model fail?
# (i) ceiling: fraction of total variance (about per-station means) inside the beat band
mu_all = kappa[valid].mean(axis=0)
var_tot = float(((kappa[valid] - mu_all) ** 2).mean())
frac_beat = float((kappa_bp[valid] ** 2).mean()) / var_tot
k_lp = signal.sosfiltfilt(sos_dc, kfill, axis=0)
sos_hi5b = signal.butter(3, BEAT_HI, btype="high", fs=fps, output="sos")
k_hi = signal.sosfiltfilt(sos_hi5b, kfill, axis=0)
results["variance_frac_beatband_ceiling"] = frac_beat
results["variance_frac_lowpass_lt1Hz"] = float((k_lp[valid] ** 2).mean()) / var_tot
results["variance_frac_gt5Hz"] = float((k_hi[valid] ** 2).mean()) / var_tot

# (ii) within-burst frequency drift (FM): a constant omega_b decoheres after ~1/(2*dur*sigma_f)
if_stds = [float(np.std(inst_freq[bi["a"]:bi["b"]])) for bi in burst_info]
results["within_burst_IF_std_Hz_mean"] = float(np.mean(if_stds))
results["burst_len_beats"] = [round(bi["dur"] * bi["f_dom"], 2) for bi in burst_info]

# (iii) oracle AM/FM variant: same spatial structure (A(s), global k, g) but the time
# course is tracked perfectly — phase = tail Hilbert phase, per-frame amplitude by LSQ.
# This is a DIAGNOSTIC upper bound (2 free functions of t), not the model.
kappa_orc = np.zeros_like(kfill)
s_ref = s[ref_station]
for bi in burst_info:
    a, b = bi["a"], bi["b"]
    Phi = inst_phase[a:b][:, None] - k_global * (s[None, :] - s_ref)
    F = A_env[None, :] * g_fun(Phi)
    y = kfill[a:b] - kfill[a:b].mean(axis=0)
    aa = (F * y).sum(axis=1) / np.maximum((F * F).sum(axis=1), 1e-9)
    kappa_orc[a:b] = aa[:, None] * F

def var_explained_fit(fit, mask):
    K = kappa[mask]; Fh = fit[mask]
    mu = K.mean(axis=0)
    return 1 - float(((K - Fh) ** 2).sum()) / float(((K - mu) ** 2).sum())

results["VE_oracleAMFM_burst_frames"] = var_explained_fit(kappa_orc, mask_burst)
results["VE_oracleAMFM_all_valid"] = var_explained_fit(kappa_orc, mask_all)

# (iv) is the >5 Hz content spatially coherent (real wave) or station-local (noise)?
def adj_corr(field):
    cs = [np.corrcoef(field[valid, j], field[valid, j + 1])[0, 1] for j in range(S - 1)]
    return float(np.mean(cs))

results["adjacent_station_corr_beatband"] = adj_corr(kappa_bp)
results["adjacent_station_corr_gt5Hz"] = adj_corr(k_hi)
results["k_per_burst_list"] = [round(float(x), 3) for x in k_per_burst]

# (v) how much of the beat band itself does the rigid model capture?
mb = mask_burst
r2_band = 1 - float(((kappa_bp[mb] - kappa_fit[mb]) ** 2).sum()) / float((kappa_bp[mb] ** 2).sum())
results["VE_rigid_vs_beatband_target_burst"] = r2_band

# ----------------------------------------------------------------------------- 6. residual + fresh PCA
resid = kappa - kappa_fit          # NaN on invalid frames
R = resid[valid]
mu_r = R.mean(axis=0)
U, sv, Vt = np.linalg.svd(R - mu_r, full_matrices=False)
evr_r = sv ** 2 / (sv ** 2).sum()
coeff_r = np.full((N, S), np.nan)
coeff_r[valid] = (R - mu_r) @ Vt.T
results["residual_EVR"] = [round(float(x), 4) for x in evr_r[:8]]
results["residual_n90"] = int(np.searchsorted(np.cumsum(evr_r), 0.90) + 1)
results["residual_n95"] = int(np.searchsorted(np.cumsum(evr_r), 0.95) + 1)

# alignment of residual PCs with the canonical kappa PCs: |cos| of mode angles
canon = np.load(PCA)["components"][:3]
align = np.abs(Vt[:3] @ canon.T)
results["residual_vs_canonical_pc_alignment_abscos"] = [
    [round(float(x), 3) for x in row] for row in align]

# spectral split of residual mean-square (gap-filled residual)
rfill = kfill - kappa_fit
r_lp = signal.sosfiltfilt(sos_dc, rfill, axis=0)
r_bp = signal.sosfiltfilt(sos_bp, rfill, axis=0)
sos_hi5 = signal.butter(3, BEAT_HI, btype="high", fs=fps, output="sos")
r_hi = signal.sosfiltfilt(sos_hi5, rfill, axis=0)
ms = lambda x: float((x[valid] ** 2).mean())
results["residual_ms_total"] = ms(rfill)
results["residual_ms_lowpass_lt1Hz_frac"] = ms(r_lp) / ms(rfill)
results["residual_ms_beatband_frac"] = ms(r_bp) / ms(rfill)
results["residual_ms_gt5Hz_frac"] = ms(r_hi) / ms(rfill)

# correlates of the top-3 residual modes
denv = np.gradient(env_s, dt)


def pearson(x, y, m=valid):
    x, y = x[m], y[m]
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum()))


mode_stats = []
for i in range(3):
    c = np.where(valid, coeff_r[:, i], np.nan)
    cf = c.copy()
    bad = ~np.isfinite(cf)
    cf[bad] = np.interp(t[bad], t[~bad], cf[~bad])
    c_bp = signal.sosfiltfilt(sos_bp, cf)
    c_am = np.abs(signal.hilbert(c_bp))
    fr, P = signal.welch(cf - cf.mean(), fs=fps, nperseg=256)
    stats = dict(
        mode=i + 1,
        corr_yaw=pearson(cf, yaw_proxy),
        corr_env=pearson(cf, env_s),
        corr_denv=pearson(cf, denv),
        corr_absAM_env=pearson(c_am, env_s),
        dom_freq_Hz=float(fr[np.argmax(P)]),
        evr=float(evr_r[i]),
    )
    mode_stats.append(stats)
results["residual_mode_stats"] = mode_stats

# ----------------------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})

# --- fig 1: burst segmentation
fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, constrained_layout=True)
ax = axes[0]
for a, b in bursts:
    ax.axvspan(t[a], t[b - 1], color=OKABE[4], alpha=0.18, lw=0)
ax.plot(t, tail, color=OKABE[0], lw=0.9, label="tail curvature (mean of last 3 stations)")
ax.set_ylabel(r"$\kappa\cdot BL$ (dimensionless)")
ax.grid(**GRID)
ax.legend(loc="upper right", frameon=False)
ax.set_title(f"Burst segmentation: {len(bursts)} bursts, "
             f"{results['burst_fraction_of_time']*100:.1f}% of time  "
             f"(band {BEAT_LO}-{BEAT_HI} Hz, thr {THR}, min {MIN_DUR}s)")
ax = axes[1]
for a, b in bursts:
    ax.axvspan(t[a], t[b - 1], color=OKABE[4], alpha=0.18, lw=0)
ax.plot(t, tail_bp, color=OKABE[0], lw=0.8, label=f"band-passed {BEAT_LO}-{BEAT_HI} Hz")
ax.plot(t, env_s, color=OKABE[1], lw=1.6, label="Hilbert envelope (2 Hz LP)")
ax.axhline(THR, color=OKABE[2], lw=1.2, ls="--", label=f"threshold {THR}")
ax.set_xlabel("time (s)")
ax.set_ylabel(r"$\kappa\cdot BL$ (dimensionless)")
ax.grid(**GRID)
ax.legend(loc="upper right", frameon=False, ncol=3)
fig.savefig(f"{OUT}/fig1_burst_segmentation.png", dpi=DPI)
plt.close(fig)

# --- fig 2: A(s)
fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
for ib in range(len(bursts)):
    ax.plot(s, absZ[ib] / absZ[ib].max() * A_raw.max(), color="0.8", lw=0.7,
            zorder=1, label="per-burst |Z(s)| (peak-scaled)" if ib == 0 else None)
ax.plot(s, A_raw, color=OKABE[0], lw=2.2, marker="o", ms=4, zorder=3,
        label="aggregated envelope A(s) (energy-weighted)")
ax.set_xlabel("s (body position, 0 = head, 1 = tail; BL)")
ax.set_ylabel(r"amplitude of fundamental  $|Z|$  ($\kappa\cdot BL$)")
ax.set_title("Amplitude envelope A(s): head recoil, node at s$\\approx$0.2, peak at s$\\approx$0.78")
ax.grid(**GRID)
ax.legend(frameon=False, loc="upper left")
fig.savefig(f"{OUT}/fig2_envelope_A.png", dpi=DPI)
plt.close(fig)

# --- fig 3: phi(s)
fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
for ib in range(len(bursts)):
    ph_b = unwrap_from_tail(np.angle(Zal[ib])); ph_b -= ph_b[ref_station]
    ax.plot(s, ph_b, color="0.85", lw=0.7, zorder=1,
            label="per-burst phase" if ib == 0 else None)
ax.plot(s, phi, color=OKABE[0], lw=2.0, marker="o", ms=4, zorder=3,
        label="aggregated phase $\\varphi(s)$")
ax.plot(s, phi_hat, color=OKABE[1], lw=1.6, ls="--", zorder=4,
        label=f"linear fit: k = {k_global:.2f} rad/BL, $R^2_w$ = {r2_phase:.3f}")
ax.plot(s, phi_q, color=OKABE[2], lw=1.4, ls=":", zorder=4,
        label=f"quadratic fit ($R^2_w$ = {r2_phase_quad:.3f})")
ax.set_xlabel("s (body position, 0 = head, 1 = tail; BL)")
ax.set_ylabel("demodulated phase (rad)")
ax.set_title(f"Phase vs body position:  $\\lambda$ = {lam:.2f} BL,  "
             f"c = {c_wave:.2f} BL/s at f = {f_global:.2f} Hz")
ax.grid(**GRID)
ax.legend(frameon=False, loc="upper right", fontsize=8)
fig.savefig(f"{OUT}/fig3_phase_phi.png", dpi=DPI)
plt.close(fig)

# --- fig 4: waveform g
fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
ax.fill_between(phase_c, g_mean - g_std, g_mean + g_std, color=OKABE[0], alpha=0.18,
                lw=0, label="$\\pm 1$ SD across folded samples")
ax.plot(phase_c, g_mean, color=OKABE[0], lw=2.0, marker="o", ms=3,
        label="folded waveform g(phase)")
ph_f = np.linspace(0, 2 * np.pi, 200)
amp1 = 2 * h1 / NBINS
ph1 = np.angle(G[1])
ax.plot(ph_f, amp1 * np.cos(ph_f + ph1), color=OKABE[1], lw=1.6, ls="--",
        label="pure sine (fundamental)")
ax.set_xlabel("beat phase (rad)")
ax.set_ylabel("normalized curvature (envelope units)")
ax.set_title(f"Waveform g: 2f power/f = {results['g_harmonic_power_2f_over_f']*100:.1f}%, "
             f"3f/f = {results['g_harmonic_power_3f_over_f']*100:.1f}%")
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.grid(**GRID)
ax.legend(frameon=False, loc="upper right", fontsize=8)
fig.savefig(f"{OUT}/fig4_waveform_g.png", dpi=DPI)
plt.close(fig)

# --- fig 5: heatmaps over an active window (longest burst +- 0.5 s)
ilong = int(np.argmax([bi["dur"] for bi in burst_info]))
a5, b5 = burst_info[ilong]["a"], burst_info[ilong]["b"]
w0 = max(0, a5 - int(0.5 * fps))
w1 = min(N, w0 + int(4.0 * fps))
vmax = np.nanpercentile(np.abs(kfill[w0:w1]), 99)
fig, axes = plt.subplots(3, 1, figsize=(9, 7.0), sharex=True, constrained_layout=True)
mats = [kfill[w0:w1].T, kappa_fit[w0:w1].T, (kfill - kappa_fit)[w0:w1].T]
names = ["observed $\\kappa\\cdot BL$", "traveling-wave fit", "residual"]
for ax, M, nm in zip(axes, mats, names):
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[t[w0], t[w1 - 1], 1, 0])
    ax.set_ylabel("s (head=0)")
    ax.text(0.008, 0.94, nm, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", alpha=0.8, ec="none", pad=1.5))
    for a, b in bursts:
        if t[min(b, N - 1)] > t[w0] and t[a] < t[w1 - 1]:
            ax.axvline(t[a], color="0.3", lw=0.7, ls=":")
            ax.axvline(t[min(b, N - 1)], color="0.3", lw=0.7, ls=":")
axes[-1].set_xlabel("time (s)")
cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.012)
cb.set_label(r"$\kappa\cdot BL$")
axes[0].set_title(f"Observed vs model vs residual ({t[w0]:.1f}-{t[w1-1]:.1f} s; "
                  "dotted lines = burst edges; model = 0 in coasts)")
fig.savefig(f"{OUT}/fig5_heatmaps.png", dpi=DPI)
plt.close(fig)

# --- fig 6: residual EVR vs canonical EVR
p = np.load(PCA)
evr_orig = p["explained_variance_ratio"]
fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
idx = np.arange(1, 9)
ax.bar(idx - 0.2, evr_orig[:8], width=0.38, color=OKABE[0], label="original kappa PCA (canonical)")
ax.bar(idx + 0.2, evr_r[:8], width=0.38, color=OKABE[1], label="residual PCA (fresh fit)")
for i in range(3):
    ax.text(idx[i] + 0.2, evr_r[i] + 0.012, f"{evr_r[i]:.2f}", ha="center", fontsize=8)
ax.set_xlabel("principal component")
ax.set_ylabel("explained variance ratio")
ax.set_title("Residual PCA nearly reproduces the canonical spectrum:\nthe rigid wave model removed only a thin slice")
ax.grid(axis="y", **GRID)
ax.legend(frameon=False)
fig.savefig(f"{OUT}/fig6_residual_evr.png", dpi=DPI)
plt.close(fig)

# --- fig 7: residual modes + correlate time series
fig, axes = plt.subplots(4, 1, figsize=(10, 9.0), constrained_layout=True)
ax = axes[0]
for i in range(3):
    ax.plot(s, Vt[i], color=OKABE[i], lw=1.8, marker="o", ms=3,
            label=f"rPC{i+1} ({evr_r[i]*100:.1f}%)")
ax.axhline(0, color="0.6", lw=0.7)
ax.set_xlabel("s (head=0, tail=1; BL)")
ax.set_ylabel("mode shape (unit norm)")
ax.set_title("Top-3 residual mode shapes")
ax.grid(**GRID)
ax.legend(frameon=False, ncol=3)


def z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x[valid])) / np.nanstd(x[valid])


correlates = [("yaw-rate proxy (LP<1Hz DC bend)", yaw_proxy, "0.35"),
              ("burst envelope", env_s, "0.35"),
              ("d(envelope)/dt", denv, "0.35")]
for i in range(3):
    ax = axes[i + 1]
    st = mode_stats[i]
    ax.plot(t, z(np.where(valid, coeff_r[:, i], np.nan)), color=OKABE[i], lw=0.9,
            label=f"rPC{i+1} coeff (z-scored)")
    nm, sig, col = correlates[0] if abs(st["corr_yaw"]) >= max(abs(st["corr_env"]), abs(st["corr_denv"])) \
        else (correlates[1] if abs(st["corr_env"]) >= abs(st["corr_denv"]) else correlates[2])
    ax.plot(t, z(sig), color=col, lw=1.1, ls="--", label=f"{nm} (z-scored)")
    ax.set_ylabel("z-score")
    ax.set_title(f"rPC{i+1}: r(yaw)={st['corr_yaw']:+.2f}, r(env)={st['corr_env']:+.2f}, "
                 f"r(denv/dt)={st['corr_denv']:+.2f}, r(|AM|,env)={st['corr_absAM_env']:+.2f}, "
                 f"dom. freq {st['dom_freq_Hz']:.2f} Hz", fontsize=9)
    ax.grid(**GRID)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
axes[-1].set_xlabel("time (s)")
fig.savefig(f"{OUT}/fig7_residual_modes.png", dpi=DPI)
plt.close(fig)

# ----------------------------------------------------------------------------- GIF: observed vs model shape
S_FINE = 200
s_f = np.linspace(0, 1, S_FINE)


def midline_from_kappa(kprof):
    kk = np.interp(s_f, s, kprof)
    ds = s_f[1] - s_f[0]
    th = np.concatenate([[0], np.cumsum(0.5 * (kk[1:] + kk[:-1]) * ds)])
    th = th - th.mean()                       # rotate mean tangent to +x
    x = np.concatenate([[0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
    y = np.concatenate([[0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)])
    return x - x.mean(), y - y.mean()


from matplotlib.animation import PillowWriter
gif_frames = range(w0, min(w1, w0 + int(3.0 * fps)), 3)      # 3 s window, 20 fps eff.
fig, ax = plt.subplots(figsize=(6, 3.4), constrained_layout=True)
writer = PillowWriter(fps=10)
with writer.saving(fig, f"{OUT}/anim_shape_fit.gif", dpi=100):
    for fr in gif_frames:
        ax.clear()
        xo, yo = midline_from_kappa(kfill[fr])
        xm, ym = midline_from_kappa(kappa_fit[fr])
        ax.plot(xo, yo, color=OKABE[0], lw=2.4, label="observed")
        ax.plot(xm, ym, color=OKABE[1], lw=2.0, ls="--", label="traveling-wave model")
        ax.plot(xo[0], yo[0], "o", color=OKABE[0], ms=7)
        state = "BURST" if burst_mask[fr] else "coast (model = 0)"
        ax.text(0.02, 0.95, f"t = {t[fr]:.2f} s   {state}", transform=ax.transAxes,
                va="top", fontsize=10)
        ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.33, 0.33)
        ax.set_aspect("equal")
        ax.set_xlabel("x (BL)"); ax.set_ylabel("y (BL)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.grid(**GRID)
        writer.grab_frame()
plt.close(fig)

# ----------------------------------------------------------------------------- save model + results
np.savez(f"{OUT}/travelwave_model.npz",
         A_env=A_env, A_raw=A_raw, s_stations=s, g_phase_centers=phase_c,
         g_mean=g_mean, g_norm=g_norm, k_rad_per_BL=k_global,
         omega_global_rad_s=omega_global, f_global_Hz=f_global,
         phi_s=phi, phi_fit=phi_hat, burst_spans=np.array([[a, b] for a, b in bursts]),
         burst_f_dom=f_dom, burst_alpha=np.array([pb["alpha"] for pb in per_burst_fit]),
         burst_phi0=np.array([pb["phi0"] for pb in per_burst_fit]),
         kappa_fit=kappa_fit.astype(np.float32),
         residual_mean=mu_r, residual_components=Vt, residual_evr=evr_r,
         residual_coeffs=coeff_r.astype(np.float32))

with open(f"{OUT}/results.json", "w") as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(results, indent=2))
