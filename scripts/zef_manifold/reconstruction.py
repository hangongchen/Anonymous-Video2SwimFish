#!/usr/bin/env python
"""Reconstruction quality of truncated PCA on the ZeF curvature dataset.

Reads ONLY the frozen files:
  demo_out/zef_manifold/curvature_dataset.npz
  demo_out/zef_manifold/pca_basis.npz
Writes ALL outputs to demo_out/zef_manifold/reconstruction/.

Outputs:
  metrics_table.md          m vs RMSE / relative RMSE / variance explained (+ high-bend split)
  metrics.json              same numbers, machine readable
  per_station_rmse.png      per-station RMSE curve for each m
  heatmaps_original_vs_recon.png  kappa(s,t) heatmaps, t in [8.2, 10.7] s
  anim_burst_recon.gif      body-shape animation, swim burst t=[8.2, 9.6] s
  anim_turn_recon.gif       body-shape animation, sharp turn frames 805-835
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


# ----------------------------------------------------------------------------- paths
ROOT = _P("${FISH_ROOT}/demo_out/zef_manifold")
OUT = os.path.join(ROOT, "reconstruction")
os.makedirs(OUT, exist_ok=True)

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]
MS = [1, 2, 3, 5, 10]          # truncation levels for metrics/heatmaps
MS_ANIM = [1, 2, 3, 5]         # panels in the animations (plus original)

# ----------------------------------------------------------------------------- load
ds = np.load(os.path.join(ROOT, "curvature_dataset.npz"))
pca = np.load(os.path.join(ROOT, "pca_basis.npz"))

kappa = ds["kappa_bl"].astype(np.float64)        # (900, 20) kappa*BL
valid = ds["valid"]                              # (900,)
t_sec = ds["t_sec"].astype(np.float64)
frame = ds["frame"]
s_st = ds["s_stations"].astype(np.float64)       # (20,)

mean = pca["mean"].astype(np.float64)            # (20,)
comps = pca["components"].astype(np.float64)     # (20, 20) rows = PCs
coeffs = pca["coeffs"].astype(np.float64)        # (900, 20), NaN on invalid rows
evr = pca["explained_variance_ratio"].astype(np.float64)

n_frames, n_st = kappa.shape
n_valid = int(valid.sum())

# ------------------------------------------------------------------- gap-free series
# 3 invalid frames (idx 364, 393, 394; t ~ 6.07-6.57 s) -> linear interpolation in
# time, per station. Used ONLY for heatmaps/animations; all RMSE stats use valid
# frames exclusively. (The plotted windows contain no invalid frame anyway.)
kappa_gf = kappa.copy()
idx = np.arange(n_frames)
for j in range(n_st):
    col = kappa_gf[:, j]
    bad = np.isnan(col)
    if bad.any():
        col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
# project the interpolated frames onto the FROZEN basis (no refit)
coeffs_gf = (kappa_gf - mean) @ comps.T          # (900, 20)

def recon_from(c, m):
    """mean + c[:, :m] @ comps[:m]"""
    return mean + c[:, :m] @ comps[:m]

# ----------------------------------------------------------------------------- metrics
kv = kappa[valid]                                # (897, 20)
cv = coeffs[valid]
centered = kv - mean
sd_centered = float(np.sqrt(np.mean(centered ** 2)))   # pooled std of centered kappa
ss_tot = float(np.sum(centered ** 2))

maxabs = np.max(np.abs(kv), axis=1)
high = maxabs > 2.0                              # high-bend frames (C-bend turns)
n_high = int(high.sum())

rows = []
per_station_rmse = {}
recon_valid = {}
for m in MS:
    r = recon_from(cv, m)
    recon_valid[m] = r
    err = kv - r
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rel = rmse / sd_centered
    varexp = 1.0 - float(np.sum(err ** 2)) / ss_tot
    cum_evr = float(np.sum(evr[:m]))
    rmse_hi = float(np.sqrt(np.mean(err[high] ** 2)))
    rmse_lo = float(np.sqrt(np.mean(err[~high] ** 2)))
    per_station_rmse[m] = np.sqrt(np.mean(err ** 2, axis=0))
    rows.append(dict(m=m, rmse=rmse, rel_rmse=rel, var_explained=varexp,
                     cum_evr=cum_evr, rmse_high_bend=rmse_hi, rmse_rest=rmse_lo,
                     ratio_high_over_rest=rmse_hi / rmse_lo))

# ----------------------------------------------------------------------------- table
hdr = (f"# Truncated-PCA reconstruction quality (ZebraFish-05 curvature)\n\n"
       f"Valid frames: {n_valid}/900. Pooled std of centered kappa*BL = "
       f"{sd_centered:.4f}. High-bend = frames with max_s |kappa*BL| > 2 "
       f"(n={n_high}); rest n={n_valid - n_high}.\n\n")
tbl = ("| m | RMSE (kappa*BL) | relative RMSE | variance explained | cum. EVR (basis) | "
       "RMSE high-bend | RMSE rest | high/rest |\n"
       "|---|---|---|---|---|---|---|---|\n")
for r in rows:
    tbl += (f"| {r['m']} | {r['rmse']:.4f} | {r['rel_rmse']:.4f} | "
            f"{r['var_explained']:.4f} | {r['cum_evr']:.4f} | "
            f"{r['rmse_high_bend']:.4f} | {r['rmse_rest']:.4f} | "
            f"{r['ratio_high_over_rest']:.3f} |\n")
with open(os.path.join(OUT, "metrics_table.md"), "w") as f:
    f.write(hdr + tbl)
print(hdr + tbl)
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(dict(n_valid=n_valid, sd_centered=sd_centered, n_high_bend=n_high,
                   high_bend_threshold=2.0, rows=rows), f, indent=2)

# ------------------------------------------------------------- per-station RMSE curve
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for i, m in enumerate(MS):
    ax.plot(s_st, per_station_rmse[m], color=OKABE[i], lw=2,
            marker="o", ms=4, label=f"m = {m}")
ax.set_xlabel("body coordinate s (0 = head, 1 = tail tip)")
ax.set_ylabel(r"per-station RMSE ($\kappa \cdot BL$, dimensionless)")
ax.set_title("Reconstruction error along the body vs number of PCs")
ax.legend(title="components kept")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_station_rmse.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------------- heatmaps
w = (t_sec >= 8.2) & (t_sec <= 10.7)
tw = t_sec[w]
orig_w = kappa_gf[w]
vmax = float(np.max(np.abs(orig_w)))
panels = [("original", orig_w)]
for m in MS:
    panels.append((f"m = {m}", recon_from(coeffs_gf[w], m)))

fig, axes = plt.subplots(len(panels), 1, figsize=(8.4, 11.5),
                         sharex=True, sharey=True)
for ax, (name, mat) in zip(axes, panels):
    im = ax.pcolormesh(tw, s_st, mat.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="nearest")
    ax.set_ylabel("s")
    ax.text(0.008, 0.93, name, transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(fc="white", alpha=0.85, ec="none", pad=2))
    ax.invert_yaxis()  # head at top
axes[-1].set_xlabel("time (s)")
fig.suptitle(r"$\kappa(s,t)\cdot BL$: original vs truncated-PCA reconstructions,"
             " t = 8.2-10.7 s", y=0.995)
cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
cbar.set_label(r"$\kappa \cdot BL$")
fig.savefig(os.path.join(OUT, "heatmaps_original_vs_recon.png"), dpi=150,
            bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ shape rendering
S_FINE = np.linspace(0.0, 1.0, 200)

def shape_from_kappa(k20):
    """Unit-length midline from a 20-station curvature profile (kappa*BL)."""
    k = np.interp(S_FINE, s_st, k20)
    ds_ = np.diff(S_FINE)
    # theta(s) = cumulative trapezoid of k ds, theta(0) = 0
    th = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * ds_)])
    cx, cy = np.cos(th), np.sin(th)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (cx[1:] + cx[:-1]) * ds_)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (cy[1:] + cy[:-1]) * ds_)])
    # rotate so the mean tangent is +x, center on the centroid
    a = np.mean(th)
    R = np.array([[np.cos(-a), -np.sin(-a)], [np.sin(-a), np.cos(-a)]])
    p = R @ np.vstack([x, y])
    p -= p.mean(axis=1, keepdims=True)
    return p[0], p[1]

def make_anim(idx_list, fname, title, gif_fps, sub=1):
    """5-panel GIF: [original | m=1 | m=2 | m=3 | m=5]; body shape on top,
    curvature profile below (reconstruction over the original in gray)."""
    idx_list = idx_list[::sub]
    ncol = 1 + len(MS_ANIM)
    kmax = float(np.max(np.abs(kappa_gf[idx_list]))) * 1.05
    frames_png = []
    fig = plt.figure(figsize=(16.0, 5.4), dpi=100)
    gs = fig.add_gridspec(2, ncol, height_ratios=[2.1, 1.0],
                          hspace=0.32, wspace=0.28,
                          left=0.045, right=0.99, top=0.86, bottom=0.10)
    names = ["original"] + [f"m = {m}" for m in MS_ANIM]
    colors = ["#333333"] + OKABE[: len(MS_ANIM)]
    ax_sh, ax_pr = [], []
    for c in range(ncol):
        a1 = fig.add_subplot(gs[0, c]); a2 = fig.add_subplot(gs[1, c])
        a1.set_xlim(-0.62, 0.62); a1.set_ylim(-0.52, 0.52)
        a1.set_aspect("equal"); a1.set_xticks([]); a1.set_yticks([])
        a1.set_title(names[c], fontsize=12, color=colors[c])
        a2.set_xlim(0, 1); a2.set_ylim(-kmax, kmax)
        a2.axhline(0, color="0.8", lw=0.8)
        a2.set_xlabel("s", fontsize=9)
        if c == 0:
            a1.set_ylabel("body shape (unit BL)")
            a2.set_ylabel(r"$\kappa \cdot BL$", fontsize=9)
        else:
            a2.set_yticklabels([])
        a2.tick_params(labelsize=8)
        ax_sh.append(a1); ax_pr.append(a2)
    ttl = fig.suptitle("", fontsize=13)

    ln_sh, pt_hd, ln_pr, ln_pr_org = [], [], [], []
    for c in range(ncol):
        (l1,) = ax_sh[c].plot([], [], color=colors[c], lw=3, solid_capstyle="round")
        (p1,) = ax_sh[c].plot([], [], "o", color=colors[c], ms=7)
        lo = None
        if c > 0:
            (lo,) = ax_pr[c].plot([], [], color="0.7", lw=1.6, label="original")
        (l2,) = ax_pr[c].plot([], [], color=colors[c], lw=1.8,
                              label=names[c] if c > 0 else None)
        if c > 0:
            ax_pr[c].legend(fontsize=7, loc="upper right", frameon=False)
        ln_sh.append(l1); pt_hd.append(p1); ln_pr.append(l2); ln_pr_org.append(lo)

    for i in idx_list:
        ks = [kappa_gf[i]] + [recon_from(coeffs_gf[i:i + 1], m)[0] for m in MS_ANIM]
        for c in range(ncol):
            x, y = shape_from_kappa(ks[c])
            ln_sh[c].set_data(x, y)
            pt_hd[c].set_data([x[0]], [y[0]])
            ln_pr[c].set_data(s_st, ks[c])
            if c > 0:
                ln_pr_org[c].set_data(s_st, ks[0])
        ttl.set_text(f"{title}   |   t = {t_sec[i]:6.3f} s   (frame {int(frame[i])})")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames_png.append(buf)
    plt.close(fig)
    path = os.path.join(OUT, fname)
    imageio.mimsave(path, frames_png, fps=gif_fps, loop=0)
    print(f"{fname}: {len(frames_png)} frames, "
          f"{os.path.getsize(path)/1e6:.2f} MB")
    return len(frames_png)

# (a) swim burst t = [8.2, 9.6] s -> 85 data frames, keep all
ia = np.where((t_sec >= 8.2) & (t_sec <= 9.6))[0]
na = make_anim(list(ia), "anim_burst_recon.gif",
               "swim burst, original vs PCA truncations", gif_fps=18)
# (b) sharp C-bend turn, frames 805-835 (idx 804..834) -> 31 frames
ib = np.where((frame >= 805) & (frame <= 835))[0]
nb = make_anim(list(ib), "anim_turn_recon.gif",
               "sharp turn (C-bend), original vs PCA truncations", gif_fps=10)

print("anim frames:", na, nb)
print("done ->", OUT)
