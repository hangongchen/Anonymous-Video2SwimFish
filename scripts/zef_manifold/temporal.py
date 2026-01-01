#!/usr/bin/env python
"""Temporal analysis of the canonical PCA coefficients of the ZebraFish-05
curvature dataset (yourname=temporal).

Inputs (frozen, read-only):
  demo_out/zef_manifold/pca_basis.npz      -> coeffs (900,20), NaN rows invalid
  demo_out/zef_manifold/curvature_dataset.npz (fps check only)

Outputs -> demo_out/zef_manifold/temporal/
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


# ----------------------------------------------------------------------------
ROOT = _P("${FISH_ROOT}/demo_out/zef_manifold")
OUT = os.path.join(ROOT, "temporal")
os.makedirs(OUT, exist_ok=True)

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 150, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5,
})

FS = 60.0
results = {}

# ----------------------------------------------------------------------------
# Load + gap-fill the 3 NaN frames by linear interpolation (stated in report)
pca = np.load(os.path.join(ROOT, "pca_basis.npz"))
coeffs = pca["coeffs"].astype(np.float64)          # (900, 20)
n = coeffs.shape[0]
t = np.arange(n) / FS
nan_rows = np.where(np.isnan(coeffs[:, 0]))[0]
results["nan_rows_interpolated"] = nan_rows.tolist()
good = ~np.isnan(coeffs[:, 0])
A = np.empty((n, 4))
for i in range(4):
    A[:, i] = np.interp(t, t[good], coeffs[good, i])
a1, a2, a3, a4 = A.T

# ----------------------------------------------------------------------------
# Helpers
def bandpass(x, f_lo, f_hi, order=4):
    sos = signal.butter(order, [f_lo, f_hi], btype="bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x)

def welch(x, nperseg=256):
    # 256-sample (4.27 s) Hann windows, 50% overlap: 0.234 Hz resolution,
    # 6 averaged segments -- a compromise for this non-stationary 15-s record.
    return signal.welch(x, fs=FS, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                        detrend="constant")

def psd_peak(f, P, f_lo, f_hi):
    m = (f >= f_lo) & (f <= f_hi)
    i = np.argmax(P[m])
    return float(f[m][i]), float(P[m][i])

def circ_stats(dphi):
    z = np.mean(np.exp(1j * dphi))
    return float(np.abs(z)), float(np.degrees(np.angle(z)))

# ----------------------------------------------------------------------------
# Beat band + burst detection.
# Beat band chosen as 1.5-5 Hz: brackets the known 2-4 Hz burst tail beats
# while excluding the slow C-bend/turn content (<1 Hz).
BAND = (1.5, 5.0)
a1f = bandpass(a1, *BAND)
a2f = bandpass(a2, *BAND)

h1 = signal.hilbert(a1f)
h2 = signal.hilbert(a2f)
env12 = np.sqrt(np.abs(h1) ** 2 + np.abs(h2) ** 2)
# smooth envelope with a 0.25 s moving average
k = int(0.25 * FS)
env_s = np.convolve(env12, np.ones(k) / k, mode="same")

# Otsu threshold on the smoothed envelope (bimodal burst/coast distribution)
def otsu_threshold(x, nbins=128):
    hist, edges = np.histogram(x, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(float) / hist.sum()
    best, thr = -1.0, centers[0]
    for j in range(1, nbins - 1):
        w0, w1 = w[:j].sum(), w[j:].sum()
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        m0 = (w[:j] * centers[:j]).sum() / w0
        m1 = (w[j:] * centers[j:]).sum() / w1
        var_b = w0 * w1 * (m0 - m1) ** 2
        if var_b > best:
            best, thr = var_b, centers[j]
    return thr

thr = otsu_threshold(env_s)
burst = env_s > thr
# drop burst/coast segments shorter than 0.15 s (morphological clean-up)
min_len = int(0.15 * FS)
def clean_mask(m, min_len):
    m = m.copy()
    edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s < min_len:
            m[s:e] = False
    return m
burst = clean_mask(burst, min_len)
burst_frac = float(burst.mean())
n_bursts = int(np.sum(np.diff(np.r_[0, burst.astype(int)]) == 1))
results["beat_band_hz"] = list(BAND)
results["burst_threshold_env"] = float(thr)
results["burst_fraction_of_time"] = burst_frac
results["n_burst_events"] = n_bursts

# zoom window: 2.5 s centered on the strongest burst
ic = int(np.argmax(env_s))
half = int(1.25 * FS)
z0 = max(0, min(ic - half, n - 2 * half))
z1 = z0 + 2 * half
results["zoom_window_s"] = [float(t[z0]), float(t[z1 - 1])]

# ----------------------------------------------------------------------------
# 1) time series: full record + zoom
def shade_bursts(ax):
    edges = np.flatnonzero(np.diff(np.r_[0, burst.astype(int), 0]))
    for s, e in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[s], t[min(e, n - 1)], color="0.85", zorder=0, lw=0)

labels = [r"$a_1$ (PC1)", r"$a_2$ (PC2)", r"$a_3$ (PC3)", r"$a_4$ (PC4)"]
fig, axes = plt.subplots(4, 1, figsize=(10, 7.2), sharex=True)
for i, ax in enumerate(axes):
    shade_bursts(ax)
    ax.plot(t, A[:, i], color=OKABE[i], lw=0.9, label=labels[i])
    ax.set_ylabel(f"{labels[i]}\n" + r"[$\kappa\,\mathrm{BL}$]")
    ax.legend(loc="upper right", frameon=False)
axes[0].set_title("PCA coefficients $a_1..a_4$, full 15 s record "
                  "(gray = detected tail-beat bursts)")
axes[-1].set_xlabel("time [s]")
fig.align_ylabels(axes)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "timeseries_full.png"))
plt.close(fig)

fig, axes = plt.subplots(4, 1, figsize=(9, 7.2), sharex=True)
for i, ax in enumerate(axes):
    shade_bursts(ax)
    ax.plot(t, A[:, i], color=OKABE[i], lw=1.3, label=labels[i])
    ax.set_xlim(t[z0], t[z1 - 1])
    seg = A[z0:z1, i]
    pad = 0.1 * (seg.max() - seg.min() + 1e-9)
    ax.set_ylim(seg.min() - pad, seg.max() + pad)
    ax.set_ylabel(f"{labels[i]}\n" + r"[$\kappa\,\mathrm{BL}$]")
    ax.legend(loc="upper right", frameon=False)
axes[0].set_title(f"2.5-s zoom on the strongest burst "
                  f"({t[z0]:.2f}-{t[z1-1]:.2f} s)")
axes[-1].set_xlabel("time [s]")
fig.align_ylabels(axes)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "timeseries_zoom.png"))
plt.close(fig)

# ----------------------------------------------------------------------------
# 2) Welch PSDs + dominant frequencies
fig, ax = plt.subplots(figsize=(8, 5))
dom_all, dom_beat = [], []
for i in range(4):
    f, P = welch(A[:, i])
    fd_all, _ = psd_peak(f, P, 0.2, 29.9)
    fd_beat, _ = psd_peak(f, P, 1.0, 8.0)
    dom_all.append(fd_all)
    dom_beat.append(fd_beat)
    ax.semilogy(f, P, color=OKABE[i], lw=1.4,
                label=f"{labels[i]}  (peak {fd_all:.2f} Hz, beat-band {fd_beat:.2f} Hz)")
    ax.axvline(fd_beat, color=OKABE[i], lw=0.7, ls=":", alpha=0.7)
ax.axvspan(*BAND, color="0.9", zorder=0)
ax.set_xlim(0, 15)
ax.set_xlabel("frequency [Hz]")
ax.set_ylabel(r"PSD [$(\kappa\,\mathrm{BL})^2$/Hz]")
ax.set_title("Welch PSD (4.27-s Hann windows, 50% overlap; gray = 1.5-5 Hz beat band)")
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "psd_welch.png"))
plt.close(fig)
results["dominant_freq_hz_full_record"] = {f"a{i+1}": round(dom_all[i], 4) for i in range(4)}
results["dominant_freq_hz_beat_band_1_8"] = {f"a{i+1}": round(dom_beat[i], 4) for i in range(4)}
results["harmonic_ratio_f3_over_f1_beatband"] = round(dom_beat[2] / dom_beat[0], 4)
results["harmonic_ratio_f4_over_f1_beatband"] = round(dom_beat[3] / dom_beat[0], 4)

# ----------------------------------------------------------------------------
# 2b) STFT spectrograms of a1 and a2 (~0.7 s window)
nper = 42                      # 42 samples = 0.70 s
nov = 36                       # 86% overlap -> 0.1 s hop
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
stft_store = {}
for i, (x, ax) in enumerate(zip([a1, a2], axes)):
    fS, tS, S = signal.spectrogram(x, fs=FS, window="hann", nperseg=nper,
                                   noverlap=nov, detrend="constant",
                                   mode="magnitude")
    stft_store[i] = (fS, tS, S)
    db = 20 * np.log10(S + 1e-6)
    pc = ax.pcolormesh(tS, fS, db, shading="auto", cmap="magma",
                       vmin=db.max() - 45, vmax=db.max())
    ax.axhline(BAND[0], color="w", lw=0.7, ls="--", alpha=0.8)
    ax.axhline(BAND[1], color="w", lw=0.7, ls="--", alpha=0.8)
    # burst intervals as ticks on top
    edges = np.flatnonzero(np.diff(np.r_[0, burst.astype(int), 0]))
    for s_, e_ in zip(edges[::2], edges[1::2]):
        ax.plot([t[s_], t[min(e_, n - 1)]], [27.5, 27.5], color="#56B4E9", lw=3,
                solid_capstyle="butt")
    ax.set_ylim(0, 30)
    ax.set_ylabel("frequency [Hz]")
    ax.set_title(f"STFT of {labels[i]} (0.70-s Hann window, 0.10-s hop; "
                 "dashes = 1.5-5 Hz band, blue bars = bursts)")
    cb = fig.colorbar(pc, ax=ax, pad=0.01)
    cb.set_label("magnitude [dB]")
axes[-1].set_xlabel("time [s]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "spectrogram_a1_a2.png"))
plt.close(fig)

# in-burst instantaneous beat frequency (per STFT slice peak in 1-8 Hz) for a1
fS, tS, S = stft_store[0]
burst_at_ts = np.interp(tS, t, burst.astype(float)) > 0.5
mband = (fS >= 1.0) & (fS <= 8.0)
peak_f = fS[mband][np.argmax(S[mband][:, :], axis=0)]
in_burst_f = peak_f[burst_at_ts]
results["in_burst_beat_freq_hz"] = {
    "median": round(float(np.median(in_burst_f)), 3),
    "p10": round(float(np.percentile(in_burst_f, 10)), 3),
    "p90": round(float(np.percentile(in_burst_f, 90)), 3),
    "min": round(float(in_burst_f.min()), 3),
    "max": round(float(in_burst_f.max()), 3),
}
# burst durations and tail beats per burst
edges = np.flatnonzero(np.diff(np.r_[0, burst.astype(int), 0]))
durs = (edges[1::2] - edges[::2]) / FS
f_in = float(np.median(in_burst_f))
results["burst_stats"] = {
    "mean_duration_s": round(float(durs.mean()), 4),
    "median_duration_s": round(float(np.median(durs)), 4),
    "max_duration_s": round(float(durs.max()), 4),
    "mean_beats_per_burst": round(float(durs.mean()) * f_in, 3),
}

# ----------------------------------------------------------------------------
# 3) Oscillator test PC1-PC2
phi1 = np.unwrap(np.angle(h1))
phi2 = np.unwrap(np.angle(h2))
dphi12 = np.angle(np.exp(1j * (phi1 - phi2)))     # wrapped difference
plv12, mang12 = circ_stats(dphi12)
plv12_b, mang12_b = circ_stats(dphi12[burst])
results["PC1_PC2"] = {
    "band_hz": list(BAND),
    "PLV_full": round(plv12, 4), "phase_diff_deg_full": round(mang12, 2),
    "PLV_burst_only": round(plv12_b, 4), "phase_diff_deg_burst": round(mang12_b, 2),
}
# rotation-direction consistency in the (a1f, a2f) plane
da1 = np.gradient(a1f) * FS
da2 = np.gradient(a2f) * FS
omega = (a1f * da2 - a2f * da1) / (a1f ** 2 + a2f ** 2 + 1e-12)
rot_frac_burst = float(np.mean(np.sign(omega[burst]) ==
                               np.sign(np.median(omega[burst]))))
results["PC1_PC2"]["rotation_sign_median_burst"] = float(np.sign(np.median(omega[burst])))
results["PC1_PC2"]["rotation_direction_consistency_burst"] = round(rot_frac_burst, 4)
sgn_sin = np.sign(np.sin(dphi12[burst]))
results["PC1_PC2"]["sign_sin_dphi_consistency_burst"] = round(
    float(np.mean(sgn_sin == np.sign(np.median(np.sin(dphi12[burst]))))), 4)

# phase portrait: band-passed (colored by time) + burst zoom + raw trajectory
from matplotlib.collections import LineCollection

def time_colored_line(ax, x, y, tt, lw=0.9):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis",
                        norm=plt.Normalize(tt.min(), tt.max()), linewidths=lw)
    lc.set_array(tt[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    return lc

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
lc = time_colored_line(axes[0], a1f, a2f, t)
axes[0].set_xlabel(r"$a_1$ band-passed 1.5-5 Hz [$\kappa\,\mathrm{BL}$]")
axes[0].set_ylabel(r"$a_2$ band-passed 1.5-5 Hz [$\kappa\,\mathrm{BL}$]")
axes[0].set_title(f"Band-passed portrait, full 15 s\nPLV={plv12:.3f}, "
                  f"$\\Delta\\phi$={mang12:+.1f}$^\\circ$ (burst-only "
                  f"{plv12_b:.3f}, {mang12_b:+.1f}$^\\circ$)")
axes[0].set_aspect("equal", adjustable="datalim")
cb = fig.colorbar(lc, ax=axes[0], pad=0.02); cb.set_label("time [s]")
lc = time_colored_line(axes[1], a1f[z0:z1], a2f[z0:z1], t[z0:z1], lw=1.6)
axes[1].set_xlabel(r"$a_1$ band-passed 1.5-5 Hz [$\kappa\,\mathrm{BL}$]")
axes[1].set_ylabel(r"$a_2$ band-passed 1.5-5 Hz [$\kappa\,\mathrm{BL}$]")
axes[1].set_title(f"Strongest-burst window {t[z0]:.2f}-{t[z1-1]:.2f} s\n"
                  "(rotation = traveling wave)")
axes[1].set_aspect("equal", adjustable="datalim")
cb = fig.colorbar(lc, ax=axes[1], pad=0.02); cb.set_label("time [s]")
lc = time_colored_line(axes[2], a1, a2, t, lw=0.7)
axes[2].set_xlabel(r"raw $a_1$ [$\kappa\,\mathrm{BL}$]")
axes[2].set_ylabel(r"raw $a_2$ [$\kappa\,\mathrm{BL}$]")
axes[2].set_title("Raw (unfiltered) $a_1$-$a_2$ trajectory")
axes[2].set_aspect("equal", adjustable="datalim")
cb = fig.colorbar(lc, ax=axes[2], pad=0.02); cb.set_label("time [s]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "phase_portrait_pc12.png"))
plt.close(fig)

# phase-difference time series
fig, ax = plt.subplots(figsize=(10, 3.6))
shade_bursts(ax)
ax.plot(t, np.degrees(dphi12), color=OKABE[0], lw=0.8,
        label=r"$\phi_1-\phi_2$ (wrapped)")
ax.axhline(90, color=OKABE[1], lw=0.9, ls="--", label=r"$\pm90^\circ$ quadrature")
ax.axhline(-90, color=OKABE[1], lw=0.9, ls="--")
ax.set_ylim(-185, 185)
ax.set_xlabel("time [s]")
ax.set_ylabel("phase difference [deg]")
ax.set_title("Instantaneous PC1-PC2 phase difference (gray = bursts)")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "phase_diff_pc12.png"))
plt.close(fig)

# ----------------------------------------------------------------------------
# 4) PC3-PC4 and PC1-PC3; harmonic test
a3f = bandpass(a3, *BAND)
a4f = bandpass(a4, *BAND)
h3 = signal.hilbert(a3f); h4 = signal.hilbert(a4f)
phi3 = np.unwrap(np.angle(h3)); phi4 = np.unwrap(np.angle(h4))

dphi34 = np.angle(np.exp(1j * (phi3 - phi4)))
plv34, mang34 = circ_stats(dphi34)
plv34_b, mang34_b = circ_stats(dphi34[burst])
results["PC3_PC4"] = {"band_hz": list(BAND),
    "PLV_full": round(plv34, 4), "phase_diff_deg_full": round(mang34, 2),
    "PLV_burst_only": round(plv34_b, 4), "phase_diff_deg_burst": round(mang34_b, 2)}

dphi13 = np.angle(np.exp(1j * (phi1 - phi3)))
plv13, mang13 = circ_stats(dphi13)
plv13_b, mang13_b = circ_stats(dphi13[burst])
results["PC1_PC3_1to1"] = {"band_hz": list(BAND),
    "PLV_full": round(plv13, 4), "phase_diff_deg_full": round(mang13, 2),
    "PLV_burst_only": round(plv13_b, 4), "phase_diff_deg_burst": round(mang13_b, 2)}

# 2:1 test -- a3 band-passed around twice the beat band
BAND2 = (3.0, 10.0)
a3f2 = bandpass(a3, *BAND2)
phi3_2 = np.unwrap(np.angle(signal.hilbert(a3f2)))
dphi_21 = np.angle(np.exp(1j * (2 * phi1 - phi3_2)))
plv21, mang21 = circ_stats(dphi_21)
plv21_b, mang21_b = circ_stats(dphi_21[burst])
results["PC1_PC3_2to1"] = {"a3_band_hz": list(BAND2),
    "PLV_full": round(plv21, 4), "phase_diff_deg_full": round(mang21, 2),
    "PLV_burst_only": round(plv21_b, 4), "phase_diff_deg_burst": round(mang21_b, 2)}
a4f2 = bandpass(a4, *BAND2)
phi4_2 = np.unwrap(np.angle(signal.hilbert(a4f2)))
dphi_21b = np.angle(np.exp(1j * (2 * phi1 - phi4_2)))
plv21c, mang21c = circ_stats(dphi_21b[burst])
results["PC1_PC4_2to1"] = {"a4_band_hz": list(BAND2),
    "PLV_burst_only": round(plv21c, 4), "phase_diff_deg_burst": round(mang21c, 2)}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
sc = axes[0, 0].scatter(a3f, a4f, c=t, cmap="viridis", s=4, lw=0)
axes[0, 0].set_xlabel(r"$a_3$ bp 1.5-5 Hz [$\kappa\,\mathrm{BL}$]")
axes[0, 0].set_ylabel(r"$a_4$ bp 1.5-5 Hz [$\kappa\,\mathrm{BL}$]")
axes[0, 0].set_title(f"PC3-PC4 portrait  PLV={plv34:.3f}, "
                     f"$\\Delta\\phi$={mang34:+.1f}$^\\circ$\n(burst-only "
                     f"{plv34_b:.3f}, {mang34_b:+.1f}$^\\circ$)")
axes[0, 0].set_aspect("equal", adjustable="datalim")
fig.colorbar(sc, ax=axes[0, 0], pad=0.02).set_label("time [s]")
sc = axes[0, 1].scatter(a1f, a3f, c=t, cmap="viridis", s=4, lw=0)
axes[0, 1].set_xlabel(r"$a_1$ bp 1.5-5 Hz [$\kappa\,\mathrm{BL}$]")
axes[0, 1].set_ylabel(r"$a_3$ bp 1.5-5 Hz [$\kappa\,\mathrm{BL}$]")
axes[0, 1].set_title(f"PC1-PC3 (1:1) portrait  PLV={plv13:.3f}, "
                     f"$\\Delta\\phi$={mang13:+.1f}$^\\circ$\n(burst-only "
                     f"{plv13_b:.3f}, {mang13_b:+.1f}$^\\circ$)")
axes[0, 1].set_aspect("equal", adjustable="datalim")
fig.colorbar(sc, ax=axes[0, 1], pad=0.02).set_label("time [s]")
shade_bursts(axes[1, 0])
axes[1, 0].plot(t, np.degrees(dphi34), color=OKABE[2], lw=0.8,
                label=r"$\phi_3-\phi_4$")
axes[1, 0].set_xlabel("time [s]"); axes[1, 0].set_ylabel("phase diff [deg]")
axes[1, 0].set_ylim(-185, 185)
axes[1, 0].set_title("PC3-PC4 phase difference")
axes[1, 0].legend(loc="upper right", frameon=False)
shade_bursts(axes[1, 1])
axes[1, 1].plot(t, np.degrees(dphi_21), color=OKABE[3], lw=0.8,
                label=r"$2\phi_1-\phi_3$ ($a_3$ bp 3-10 Hz)")
axes[1, 1].set_xlabel("time [s]"); axes[1, 1].set_ylabel("phase diff [deg]")
axes[1, 1].set_ylim(-185, 185)
axes[1, 1].set_title(f"2:1 test  PLV={plv21:.3f} (burst-only {plv21_b:.3f})")
axes[1, 1].legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "phase_pc34_pc13.png"))
plt.close(fig)

# ----------------------------------------------------------------------------
# 5) autocorrelations
def autocorr_biased(x, max_lag):
    x = x - x.mean()
    r = np.correlate(x, x, mode="full")[len(x) - 1:]
    r = r / r[0]
    return r[:max_lag]

max_lag = int(8 * FS)          # 8 s of lags shown/used
lag_pad = int(10 * FS)         # compute longer to avoid Hilbert edge artifact
lags = np.arange(max_lag) / FS
r_raw = autocorr_biased(a1, max_lag)
r_bp = autocorr_biased(a1f, lag_pad)
env_a1 = np.abs(h1)            # Hilbert amplitude envelope of band-passed a1
r_env = autocorr_biased(env_a1, max_lag)

# envelope of the oscillatory band-passed autocorr -> e-folding lag
r_bp_env = np.abs(signal.hilbert(r_bp))[:max_lag]
r_bp = r_bp[:max_lag]
below = np.flatnonzero(r_bp_env < (r_bp_env[0] / np.e))
efold_lag_s = float(lags[below[0]]) if below.size else float("nan")
f_beat_a1 = dom_beat[0]
efold_cycles = efold_lag_s * f_beat_a1
results["autocorr"] = {
    "a1_bp_efold_lag_s": round(efold_lag_s, 4),
    "beat_freq_used_hz": round(f_beat_a1, 4),
    "phase_coherence_cycles": round(efold_cycles, 3),
    "phase_coherence_cycles_at_inburst_freq": round(
        efold_lag_s * float(np.median(in_burst_f)), 3),
}
# envelope autocorr e-fold + first side lobe (burst recurrence)
below_e = np.flatnonzero(r_env < 1 / np.e)
results["autocorr"]["envelope_efold_lag_s"] = (
    round(float(lags[below_e[0]]), 4) if below_e.size else None)
pk, _ = signal.find_peaks(r_env[int(0.5 * FS):], height=0.1)
if pk.size:
    results["autocorr"]["envelope_first_side_peak_lag_s"] = round(
        float(lags[int(0.5 * FS) + pk[0]]), 4)
    results["autocorr"]["envelope_first_side_peak_value"] = round(
        float(r_env[int(0.5 * FS) + pk[0]]), 4)

fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
axes[0].plot(lags, r_raw, color=OKABE[0], lw=1.2, label=r"raw $a_1$")
axes[0].plot(lags, r_bp, color=OKABE[1], lw=1.2,
             label=r"$a_1$ band-passed 1.5-5 Hz")
axes[0].plot(lags, r_bp_env, color=OKABE[2], lw=1.2, ls="--",
             label="envelope of band-passed autocorr")
axes[0].axhline(1 / np.e, color="0.4", lw=0.8, ls=":", label=r"$1/e$")
if np.isfinite(efold_lag_s):
    axes[0].axvline(efold_lag_s, color="0.4", lw=0.8, ls=":")
    axes[0].annotate(f"e-fold {efold_lag_s:.2f} s\n= {efold_cycles:.1f} beat cycles",
                     xy=(efold_lag_s, 1 / np.e), xytext=(efold_lag_s + 0.4, 0.62),
                     fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))
axes[0].set_ylabel("autocorrelation")
axes[0].set_title(r"Autocorrelation of $a_1$")
axes[0].legend(frameon=False, fontsize=8.5)
axes[1].plot(lags, r_env, color=OKABE[3], lw=1.2,
             label=r"Hilbert amplitude envelope of band-passed $a_1$")
axes[1].axhline(1 / np.e, color="0.4", lw=0.8, ls=":", label=r"$1/e$")
axes[1].axhline(0, color="0.6", lw=0.6)
axes[1].set_xlabel("lag [s]")
axes[1].set_ylabel("autocorrelation")
axes[1].set_title("Autocorrelation of the beat-amplitude envelope "
                  "(burst recurrence timescale)")
axes[1].legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "autocorr_a1.png"))
plt.close(fig)

# ----------------------------------------------------------------------------
with open(os.path.join(OUT, "results.json"), "w") as fjson:
    json.dump(results, fjson, indent=2)
print(json.dumps(results, indent=2))
print("\nOutputs in", OUT)
