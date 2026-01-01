#!/usr/bin/env python
"""PCA mode visualization for the ZeF curvature manifold study (agent: modes_viz).

Reads ONLY the frozen files:
  demo_out/zef_manifold/curvature_dataset.npz
  demo_out/zef_manifold/pca_basis.npz   (canonical PCA -- never refit)

Writes figures/GIFs/metrics to demo_out/zef_manifold/modes_viz/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


# ---------------------------------------------------------------- paths / style
BASE = _P("${FISH_ROOT}/demo_out/zef_manifold")
OUT = os.path.join(BASE, "modes_viz")
os.makedirs(OUT, exist_ok=True)

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
DPI = 150
plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.size": 10,
})

# ---------------------------------------------------------------- load frozen data
d = np.load(os.path.join(BASE, "curvature_dataset.npz"))
p = np.load(os.path.join(BASE, "pca_basis.npz"))

kappa = d["kappa_bl"].astype(np.float64)          # (900,20) kappa*BL
valid = d["valid"].astype(bool)
S = d["s_stations"].astype(np.float64)            # (20,) uniform 0..1
mean = p["mean"].astype(np.float64)               # (20,)
comps = p["components"].astype(np.float64)        # (20,20) rows = PCs
evr = p["explained_variance_ratio"].astype(np.float64)
coeffs = p["coeffs"].astype(np.float64)           # (900,20), NaN on invalid rows

sigma = np.nanstd(coeffs, axis=0)                 # per-PC coefficient std
raw_mean = np.nanmean(kappa[valid], axis=0)
raw_std = np.nanstd(kappa[valid], axis=0)
mean_check = float(np.max(np.abs(raw_mean - mean)))

metrics = {
    "n_frames_valid": int(valid.sum()),
    "pca_mean_vs_raw_mean_maxabsdiff": mean_check,
    "evr_1_10": [float(v) for v in evr[:10]],
    "cum_evr_1_6": [float(v) for v in np.cumsum(evr)[:6]],
    "sigma_1_5": [float(v) for v in sigma[:5]],
}

# ---------------------------------------------------------------- shape rendering
S_FINE = np.linspace(0.0, 1.0, 200)

def fine_curv(k20):
    """Interpolate a 20-station curvature profile to the fine s grid (cubic)."""
    return CubicSpline(S, k20)(S_FINE)

def render_shape(k20):
    """Unit-length midline (x, y) from kappa*BL profile; mean tangent -> +x, head at origin."""
    kf = fine_curv(k20)
    th = cumulative_trapezoid(kf, S_FINE, initial=0.0)
    th = th - np.trapz(th, S_FINE)                # mean tangent angle -> 0
    x = cumulative_trapezoid(np.cos(th), S_FINE, initial=0.0)
    y = cumulative_trapezoid(np.sin(th), S_FINE, initial=0.0)
    return x, y

# ---------------------------------------------------------------- Fig 1: mean +- std
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.fill_between(S, raw_mean - raw_std, raw_mean + raw_std,
                color=OKABE[0], alpha=0.22, label=r"raw data $\pm 1$ std")
ax.plot(S, mean, color=OKABE[0], lw=2.0, marker="o", ms=3.5, label="mean profile")
ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
ax.set_xlabel("body coordinate  s  (0 = head, 1 = tail tip)  [BL]")
ax.set_ylabel(r"curvature  $\kappa \cdot BL$  [dimensionless]")
ax.set_title("Mean curvature profile with raw-data variability (897 valid frames)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_mean_std_profile.png"))
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: EVR
cum = np.cumsum(evr)
n90 = int(p["n90"]); n95 = int(p["n95"]); n99 = int(p["n99"])
fig, (axb, axc) = plt.subplots(1, 2, figsize=(10.2, 4.0))
pcs10 = np.arange(1, 11)
axb.bar(pcs10, evr[:10], color=OKABE[0], width=0.72)
axb.set_yscale("log")
axb.set_xticks(pcs10)
axb.set_xlabel("principal component")
axb.set_ylabel("explained variance ratio  [-]  (log scale)")
axb.set_title("Per-PC explained variance (first 10 PCs)")
for i in range(4):
    axb.text(pcs10[i], evr[i] * 1.25, f"{evr[i]:.3f}", ha="center", fontsize=8)

axc.plot(np.arange(1, 11), cum[:10], color=OKABE[1], lw=2.0, marker="o", ms=4,
         label="cumulative EVR")
for thr, n_thr, dy in [(0.90, n90, -0.045), (0.95, n95, 0.018), (0.99, n99, 0.018)]:
    axc.axhline(thr, color="0.55", lw=0.9, ls="--")
    axc.text(10.1, thr, f"{thr:.2f}", va="center", fontsize=8, color="0.35")
    axc.annotate(f"n{int(thr*100)} = {n_thr}", xy=(n_thr, cum[n_thr - 1]),
                 xytext=(n_thr + 1.2, thr + dy), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="0.3", lw=0.9))
axc.set_xticks(np.arange(1, 11))
axc.set_xlabel("number of PCs kept")
axc.set_ylabel("cumulative explained variance ratio  [-]")
axc.set_ylim(0.68, 1.01)
axc.set_title("Cumulative EVR with 0.90 / 0.95 / 0.99 thresholds")
axc.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_explained_variance.png"))
plt.close(fig)

# ---------------------------------------------------------------- Figs 3.x: per-PC panels
cmap = plt.get_cmap("RdBu_r")
lvl_mult = [-2.0, -1.0, 0.0, 1.0, 2.0]           # in units of sigma_i

for i in range(5):
    pc = i + 1
    load = comps[i]
    sig = sigma[i]
    norm = Normalize(vmin=-2.0 * sig, vmax=2.0 * sig)
    # colors keyed to the signed coefficient; midpoint of RdBu_r is white -> use black for mean
    def lvl_color(m):
        return "0.15" if m == 0.0 else cmap(0.5 + 0.23 * m)  # 0.23: stay off the white center

    fig = plt.figure(figsize=(10.8, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.28)
    axl = fig.add_subplot(gs[0, 0])
    axp = fig.add_subplot(gs[0, 1])
    axs = fig.add_subplot(gs[1, :])

    # (a) loading profile
    axl.plot(S, load, color=OKABE[0], lw=2.0, marker="o", ms=3.5)
    axl.axhline(0.0, color="0.4", lw=0.8, ls=":")
    axl.set_xlabel("body coordinate  s  [BL]")
    axl.set_ylabel("loading  component$_%d$(s)  [-]" % pc)
    axl.set_title(f"PC{pc} loading (EVR = {evr[i]*100:.2f}%)")

    # (b) mean +- 1,2 sigma curvature profiles
    for m in lvl_mult:
        prof = mean + m * sig * load
        lbl = "mean" if m == 0 else (r"$%+g\sigma_%d$" % (m, pc))
        axp.plot(S, prof, color=lvl_color(m), lw=2.2 if m == 0 else 1.6, label=lbl)
    axp.axhline(0.0, color="0.4", lw=0.8, ls=":")
    axp.set_xlabel("body coordinate  s  [BL]")
    axp.set_ylabel(r"curvature  $\kappa \cdot BL$  [-]")
    axp.set_title(rf"PC{pc}: mean $\pm 1,2\,\sigma$  ($\sigma_{pc}$ = {sig:.3f})")
    axp.legend(frameon=False, fontsize=8, ncol=2)

    # (c) body shapes, coefficient color-coded
    for m in lvl_mult:
        x, y = render_shape(mean + m * sig * load)
        axs.plot(x, y, color=lvl_color(m), lw=2.2 if m == 0 else 1.6)
        if m == lvl_mult[0]:
            axs.plot(x[0], y[0], "o", color="0.15", ms=5, zorder=5)
            axs.annotate("head", xy=(x[0], y[0]), xytext=(x[0] - 0.02, y[0] + 0.06),
                         fontsize=8, color="0.3")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axs, shrink=0.9, pad=0.015)
    cb.set_label(rf"$a_{pc}$  [$\kappa\!\cdot\!BL$]")
    axs.set_aspect("equal")
    ymax = max(0.10, 1.25 * max(np.max(np.abs(render_shape(mean + m * sig * load)[1]))
                                for m in (lvl_mult[0], lvl_mult[-1])))
    axs.set_xlim(-0.04, 1.05)
    axs.set_ylim(-ymax, ymax)
    axs.set_xlabel("x  [BL]")
    axs.set_ylabel("y  [BL]")
    axs.set_title(f"PC{pc} body shapes, $a_{pc}\\in[-2\\sigma_{pc},+2\\sigma_{pc}]$ "
                  "(unit-length midlines, equal aspect)")
    fig.savefig(os.path.join(OUT, f"fig3_pc{pc}_loading_profiles_shapes.png"),
                bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- combined loadings (PC1-5)
fig, ax = plt.subplots(figsize=(6.8, 4.2))
for i in range(5):
    ax.plot(S, comps[i], color=OKABE[i], lw=1.8, marker="o", ms=3,
            label=f"PC{i+1} ({evr[i]*100:.1f}%)")
ax.axhline(0.0, color="0.4", lw=0.8, ls=":")
ax.set_xlabel("body coordinate  s  [BL]")
ax.set_ylabel("loading  [-]")
ax.set_title("PC1-PC5 loading profiles")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_loadings_pc1_5.png"))
plt.close(fig)

# ---------------------------------------------------------------- GIFs: PCs 1-4
N_PHASE = 48
for i in range(4):
    pc = i + 1
    load = comps[i]
    sig = sigma[i]
    phases = np.linspace(0.0, 2.0 * np.pi, N_PHASE, endpoint=False)
    amps = 2.0 * sig * np.sin(phases)
    shapes = [render_shape(mean + a * load) for a in amps]
    allx = np.concatenate([s[0] for s in shapes])
    ally = np.concatenate([s[1] for s in shapes])
    mx = 0.06
    xlim = (allx.min() - mx, allx.max() + mx)
    ylim = (ally.min() - mx, ally.max() + mx)

    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=130)
    fig.subplots_adjust(left=0.16, bottom=0.13, right=0.96, top=0.88)
    ln, = ax.plot([], [], color=OKABE[0], lw=2.6)
    hd, = ax.plot([], [], "o", color="0.15", ms=6)
    txt = ax.text(0.02, 0.965, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x  [BL]"); ax.set_ylabel("y  [BL]")
    ax.set_title(f"PC{pc} shape sweep:  $a_{pc} = 2\\sigma_{pc}\\sin(\\phi)$")

    def update(f, shapes=shapes, amps=amps, phases=phases, ln=ln, hd=hd, txt=txt, pc=pc):
        x, y = shapes[f]
        ln.set_data(x, y)
        hd.set_data([x[0]], [y[0]])
        txt.set_text(f"PC{pc}   $a_{pc}$ = {amps[f]:+.2f}   $\\phi$ = {phases[f]:.2f} rad")
        return ln, hd, txt

    ani = animation.FuncAnimation(fig, update, frames=N_PHASE, blit=True)
    ani.save(os.path.join(OUT, f"pc{pc}_sweep.gif"),
             writer=animation.PillowWriter(fps=16))
    plt.close(fig)

# ---------------------------------------------------------------- quantitative interpretation
NF = 512          # fine spatial samples for FFT
NFFT = 8192
sf = np.linspace(0.0, 1.0, NF)
freqs = np.fft.rfftfreq(NFFT, d=sf[1] - sf[0])   # cycles / BL

def spatial_stats(load):
    lf = CubicSpline(S, load)(sf)
    lf0 = lf - lf.mean()
    spec = np.abs(np.fft.rfft(lf0, n=NFFT))
    k = np.argmax(spec[1:]) + 1                   # skip DC
    f_dom = freqs[k]
    lam = 1.0 / f_dom
    zc20 = int(np.sum(np.diff(np.sign(load)) != 0))          # on the 20 stations
    zc_pos = sf[:-1][np.diff(np.sign(lf)) != 0]              # crossing locations (fine)
    if len(zc_pos) >= 2:
        lam_zc = 2.0 * float(np.mean(np.diff(zc_pos)))
    else:
        lam_zc = float("nan")
    tail_energy = float(np.sum(load[S > 0.5] ** 2))          # loadings are unit-norm
    return dict(dom_freq_cyc_per_BL=float(f_dom), wavelength_BL=float(lam),
                zero_crossings_20sta=zc20, wavelength_zc_BL=lam_zc,
                tail_half_energy_frac=tail_energy), lf0, spec

stats, fine_loads, specs = [], [], []
for i in range(5):
    st, lf0, spec = spatial_stats(comps[i])
    stats.append(st); fine_loads.append(lf0); specs.append(spec)
metrics["pc_spatial_stats"] = {f"PC{i+1}": stats[i] for i in range(5)}

# PC1-PC2 quadrature test. IMPORTANT: use the RAW loadings (no mean removal) --
# the loadings are exactly orthogonal on the 20 stations (dot = 3e-10) and their DC
# part is a real uniform-bend component; mean removal would fake anti-correlation.
r1 = CubicSpline(S, comps[0])(sf)
r2 = CubicSpline(S, comps[1])(sf)
ds = sf[1] - sf[0]
maxlag = NF // 2                                  # shifts up to +-0.5 BL (>=0.5 BL overlap)
lags = np.arange(-maxlag, maxlag + 1)

def shift_corr(y1, y2):
    out = np.empty(lags.size)
    for j, L in enumerate(lags):
        if L >= 0:
            a, b = y1[L:], y2[:NF - L]
        else:
            a, b = y1[:NF + L], y2[-L:]
        out[j] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    return out

cc = shift_corr(r1, r2)                           # raw loadings (DC kept)
jbest = int(np.argmax(np.abs(cc)))
best_lag_BL = float(lags[jbest] * ds)
best_corr = float(cc[jbest])
at_boundary = jbest in (0, lags.size - 1)
zero_lag_corr = float(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
# mean-removed variant: compares the OSCILLATORY parts only (envelope + DC ignored)
cc0 = shift_corr(r1 - r1.mean(), r2 - r2.mean())
jb0 = int(np.argmax(np.abs(cc0)))

# phase per loading from a DC + sinusoid least-squares fit at a common spatial freq
f1 = stats[0]["dom_freq_cyc_per_BL"]; f2 = stats[1]["dom_freq_cyc_per_BL"]
f_common = 0.5 * (f1 + f2)
def fit_phase(y, f):
    A = np.column_stack([np.ones_like(sf), np.cos(2*np.pi*f*sf), np.sin(2*np.pi*f*sf)])
    c0, cb, cs = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ np.array([c0, cb, cs])
    r2fit = 1.0 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    return np.arctan2(-cs, cb), float(np.hypot(cb, cs)), float(r2fit)  # y ~ amp*cos(2pi f s + phase) + DC
ph1, amp1, r2_1 = fit_phase(r1, f_common)
ph2, amp2, r2_2 = fit_phase(r2, f_common)
dphi = np.rad2deg(np.angle(np.exp(1j * (ph2 - ph1))))
shift_equiv_BL = (np.deg2rad(dphi) / (2*np.pi*f_common))   # spatial shift equivalent

metrics["pc1_pc2_quadrature"] = {
    "raw_zero_lag_corr": zero_lag_corr,
    "best_shift_corr": best_corr,
    "best_shift_BL": best_lag_BL,
    "best_shift_hit_lag_boundary": bool(at_boundary),
    "meanremoved_best_shift_corr": float(cc0[jb0]),
    "meanremoved_best_shift_BL": float(lags[jb0] * ds),
    "quarter_wavelength_PC1_BL": stats[0]["wavelength_BL"] / 4.0,
    "sinefit_common_freq_cyc_per_BL": float(f_common),
    "sinefit_phase_PC1_deg": float(np.rad2deg(ph1)),
    "sinefit_phase_PC2_deg": float(np.rad2deg(ph2)),
    "sinefit_phase_diff_PC2_minus_PC1_deg": float(dphi),
    "sinefit_phase_diff_shift_equiv_BL": float(shift_equiv_BL),
    "sinefit_R2_PC1": r2_1, "sinefit_R2_PC2": r2_2,
    "pc1_domfreq_cyc_per_BL": float(f1),
    "pc2_domfreq_cyc_per_BL": float(f2),
}

# node / peak landmarks of the first two loadings
pc1_peak_s = float(sf[np.argmax(np.abs(r1))])
zc2 = sf[:-1][np.diff(np.sign(r2)) != 0]
metrics["landmarks"] = {
    "pc1_peak_s_BL": pc1_peak_s,
    "pc2_node_s_BL": float(zc2[0]) if len(zc2) else None,
    "pc1_loading_mean": float(comps[0].mean()),
    "pc2_loading_mean": float(comps[1].mean()),
}

# spatial spectra figure
fig, ax = plt.subplots(figsize=(6.8, 4.2))
fmax = 6.0
sel = freqs <= fmax
for i in range(5):
    sp = specs[i][sel] / specs[i].max()
    ax.plot(freqs[sel], sp, color=OKABE[i], lw=1.8, label=f"PC{i+1}")
ax.set_xlabel("spatial frequency  [cycles / BL]")
ax.set_ylabel("normalized |FFT| of loading  [-]")
ax.set_title("Spatial spectra of PC loadings (peak = dominant wavelength)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_loading_spatial_spectra.png"))
plt.close(fig)

def _san(o):
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_san(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o

metrics = _san(metrics)
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print("done ->", OUT)
