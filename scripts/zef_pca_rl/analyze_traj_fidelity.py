"""Locomotion/displacement fidelity analysis: real ZeF vs PCA+RL vs CPG+RL.

Everything is compared in the fish's INITIAL body frame, body-length normalized:
    d_body(t) = R(-psi_0) @ (p(t) - p(t_0)) / BL          (2D top-view, both sources)
Two alignments, reported separately (speed differences must not hide in path metrics):
    TIME  : both trajectories resampled to 101 points on normalized time
    PATH  : both resampled to 101 points on normalized arc length (tau)
Chamfer distance = symmetric mean nearest-neighbor distance between the two point sets.

Inputs : demo_out/trajectory_fidelity/{zef_reference,rollout_pca,rollout_cpg}.npz
Outputs: metrics.json, REPORT.md, plot subfolders under demo_out/trajectory_fidelity/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


OUT = Path(_P("${FISH_ROOT}/demo_out/trajectory_fidelity"))
Z = np.load(OUT / "zef_reference.npz")
R = {m: np.load(OUT / f"rollout_{m}.npz") for m in ("pca", "cpg")}
S = Z["samples"]
N = S.shape[0]
HN = [str(x) for x in Z["horizon_names"]]
fpsz = float(Z["fps"])
GRID = np.linspace(0, 1, 101)


def body_frame(p, psi0, p0):
    d = p - p0
    c, s = np.cos(-psi0), np.sin(-psi0)
    return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)


def resample_time(traj):
    t = np.linspace(0, 1, len(traj))
    return np.stack([np.interp(GRID, t, traj[:, k]) for k in range(traj.shape[1])], 1)


def resample_arc(traj):
    seg = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    if cum[-1] < 1e-9:
        return np.repeat(traj[:1], len(GRID), 0)
    return np.stack([np.interp(GRID * cum[-1], cum, traj[:, k]) for k in range(traj.shape[1])], 1)


def chamfer(A, B):
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    return 0.5 * (D.min(1).mean() + D.min(0).mean())


# ---------------- per-sample trajectories ----------------
zef = []
for ti, tj, h in S:
    seg = Z["p_bl"][ti:tj + 1].astype(np.float64)
    psi = np.unwrap(Z["psi"][ti:tj + 1].astype(np.float64))
    zef.append({"d": body_frame(seg, Z["psi"][ti], seg[0]), "dpsi": psi - psi[0],
                "t": np.arange(len(seg)) / fpsz})

ag = {m: [] for m in R}
for m, r in R.items():
    dt = float(r["dt"]); BL = float(r["body_length"])
    for e in range(N):
        T = int(r["end_step"][e]) + 1
        p = r["pos"][:T, e, 0:2].astype(np.float64)
        psi = np.unwrap(r["psi"][:T, e].astype(np.float64))
        ag[m].append({"d": body_frame(p, float(r["psi0"][e]), p[0]) / BL,
                      "dpsi": psi - psi[0], "t": np.arange(T) * dt,
                      "success": int(r["outcome"][e]) == 1, "outcome": int(r["outcome"][e])})

# ---------------- per-sample metrics ----------------
rows = {m: [] for m in R}
for m in R:
    for i in range(N):
        zd, ad = zef[i]["d"], ag[m][i]["d"]
        zt, at_ = zef[i]["t"], ag[m][i]["t"]
        ztime, atime = resample_time(zd), resample_time(ad)
        zarc, aarc = resample_arc(zd), resample_arc(ad)
        Lz = float(np.sum(np.linalg.norm(np.diff(zd, axis=0), axis=1)))
        La = float(np.sum(np.linalg.norm(np.diff(ad, axis=0), axis=1)))
        vz = np.linalg.norm(np.diff(zd, axis=0), axis=1) * fpsz            # |v| BL/s
        va = np.linalg.norm(np.diff(ad, axis=0), axis=1) / max(at_[1] - at_[0], 1e-9)
        # forward velocity along instantaneous heading
        vz_f = (np.diff(zd, axis=0) * np.stack([np.cos(zef[i]["dpsi"][:-1]),
                                                np.sin(zef[i]["dpsi"][:-1])], 1)).sum(1) * fpsz
        va_f = (np.diff(ad, axis=0) * np.stack([np.cos(ag[m][i]["dpsi"][:-1]),
                                                np.sin(ag[m][i]["dpsi"][:-1])], 1)).sum(1) / max(at_[1] - at_[0], 1e-9)
        zpsi_t = np.interp(GRID, np.linspace(0, 1, len(zef[i]["dpsi"])), zef[i]["dpsi"])
        apsi_t = np.interp(GRID, np.linspace(0, 1, len(ag[m][i]["dpsi"])), ag[m][i]["dpsi"])
        eff_z = float(np.linalg.norm(zd[-1]) / max(Lz, 1e-9))
        eff_a = float(np.linalg.norm(ad[-1]) / max(La, 1e-9))
        rows[m].append({
            "horizon": int(S[i, 2]), "success": ag[m][i]["success"], "outcome": ag[m][i]["outcome"],
            "cd_time": chamfer(atime, ztime), "cd_path": chamfer(aarc, zarc),
            "dx_err": abs(ad[-1, 0] - zd[-1, 0]), "dy_err": abs(ad[-1, 1] - zd[-1, 1]),
            "dx_a": ad[-1, 0], "dy_a": ad[-1, 1], "dx_z": zd[-1, 0], "dy_z": zd[-1, 1],
            "total_disp_a": float(np.linalg.norm(ad[-1])), "total_disp_z": float(np.linalg.norm(zd[-1])),
            "L_a": La, "L_z": Lz, "L_err": abs(La - Lz),
            "eff_a": eff_a, "eff_z": eff_z, "eff_err": abs(eff_a - eff_z),
            "v_mean_a": float(va.mean()) if len(va) else 0.0, "v_mean_z": float(vz.mean()),
            "vf_mean_a": float(va_f.mean()) if len(va_f) else 0.0, "vf_mean_z": float(vz_f.mean()),
            "vf_med_a": float(np.median(va_f)) if len(va_f) else 0.0, "vf_med_z": float(np.median(vz_f)),
            "v_err_norm": abs(float(va.mean()) - float(vz.mean())) / max(float(vz.mean()), 1e-9),
            "psi_rmse": float(np.sqrt(((apsi_t - zpsi_t) ** 2).mean())),
            "psi_final_err": abs(float(apsi_t[-1] - zpsi_t[-1])),
            "psi_final_a": float(apsi_t[-1]), "psi_final_z": float(zpsi_t[-1]),
            "psi_max_a": float(np.abs(apsi_t).max()), "psi_max_z": float(np.abs(zpsi_t).max()),
            "turn_dir_agree": int(np.sign(apsi_t[-1]) == np.sign(zpsi_t[-1]))
            if abs(zpsi_t[-1]) > np.deg2rad(10) else None,
            "dur_a": float(at_[-1]), "dur_z": float(zt[-1]),
            "tgt_disp": float(np.linalg.norm(zd[-1])),
        })


def agg(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return None
    boot = np.random.default_rng(0).choice(v, (2000, len(v))).mean(1)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75)),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "n": int(len(v))}


KEYS = ["cd_time", "cd_path", "dx_err", "dy_err", "L_err", "eff_err", "v_err_norm",
        "psi_rmse", "psi_final_err"]
metrics = {"n_samples": N, "horizon_names": HN}
for m in R:
    mm = {"all": {k: agg([r[k] for r in rows[m]]) for k in KEYS}}
    mm["all"]["task_success_rate"] = float(np.mean([r["success"] for r in rows[m]]))
    mm["all"]["blowups"] = int(sum(1 for r in rows[m] if r["outcome"] == -1))
    mm["all"]["turn_dir_agreement"] = agg([r["turn_dir_agree"] for r in rows[m]])
    for hi, hn in enumerate(HN):
        sub = [r for r in rows[m] if r["horizon"] == hi]
        mm[hn] = {k: agg([r[k] for r in sub]) for k in KEYS}
        mm[hn]["task_success_rate"] = float(np.mean([r["success"] for r in sub]))
    metrics[m] = mm
# ZeF reference stats
metrics["zef_reference"] = {
    "vf_mean": agg([r["vf_mean_z"] for r in rows["pca"]]),
    "eff": agg([r["eff_z"] for r in rows["pca"]]),
    "L": agg([r["L_z"] for r in rows["pca"]]),
    "dur_s": agg([r["dur_z"] for r in rows["pca"]]),
    "psi_final": agg([abs(r["psi_final_z"]) for r in rows["pca"]]),
}
# paired PCA - CPG differences
metrics["paired_pca_minus_cpg"] = {
    k: agg([rows["pca"][i][k] - rows["cpg"][i][k] for i in range(N)]) for k in KEYS}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# ---------------- plots ----------------
for sub in ("trajectory_overlays", "displacement_plots", "velocity_plots", "heading_plots",
            "chamfer_distributions"):
    (OUT / sub).mkdir(exist_ok=True)
COL = {"zef": "k", "pca": "#1f77b4", "cpg": "#d62728"}

# overlay grids per horizon
for hi, hn in enumerate(HN):
    idx = [i for i in range(N) if S[i, 2] == hi][:12]
    fig, axs = plt.subplots(3, 4, figsize=(16, 12))
    for k, i in enumerate(idx):
        ax = axs[k // 4][k % 4]
        ax.plot(zef[i]["d"][:, 0], zef[i]["d"][:, 1], "-", color=COL["zef"], lw=2, label="ZeF")
        for m in ("pca", "cpg"):
            suffix = "" if ag[m][i]["success"] else " (fail)"
            ax.plot(ag[m][i]["d"][:, 0], ag[m][i]["d"][:, 1], "-", color=COL[m],
                    lw=1.4, label=f"{m.upper()}{suffix}")
        ax.plot(0, 0, "go", ms=8)
        ax.plot(zef[i]["d"][-1, 0], zef[i]["d"][-1, 1], "r*", ms=16)
        ax.add_patch(plt.Circle((zef[i]["d"][-1, 0], zef[i]["d"][-1, 1]), 0.2 / 0.496 * 0.496,
                                fill=False, ls="--", color="r", lw=0.8))
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"s{i} d={rows['pca'][i]['tgt_disp']:.1f}BL", fontsize=9)
        if k == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"body-frame trajectories /BL — {hn} horizon (star=target, green=start)")
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_overlays" / f"grid_{hn}.png", dpi=110)
    plt.close(fig)

# displacement + heading + velocity vs normalized time (mean +/- IQR per horizon)
for hi, hn in enumerate(HN):
    idx = [i for i in range(N) if S[i, 2] == hi]
    figs = {name: plt.subplots(1, 2, figsize=(12, 4)) for name in ("disp", "psi")}
    figv, axv = plt.subplots(1, 2, figsize=(12, 4))
    for src, col in (("zef", COL["zef"]), ("pca", COL["pca"]), ("cpg", COL["cpg"])):
        trajs = [resample_time(zef[i]["d"]) if src == "zef" else resample_time(ag[src][i]["d"]) for i in idx]
        Tm = np.stack(trajs)
        for a, k, lab in ((figs["disp"][1][0], 0, "forward /BL"), (figs["disp"][1][1], 1, "lateral /BL")):
            med = np.median(Tm[:, :, k], 0)
            a.plot(GRID, med, color=col, label=src.upper())
            a.fill_between(GRID, np.percentile(Tm[:, :, k], 25, 0), np.percentile(Tm[:, :, k], 75, 0),
                           color=col, alpha=0.15)
            a.set_title(lab); a.legend(fontsize=8); a.grid(alpha=0.3); a.set_xlabel("normalized time")
        ps = np.stack([np.interp(GRID, np.linspace(0, 1, len(
            (zef[i] if src == "zef" else ag[src][i])["dpsi"])),
            (zef[i] if src == "zef" else ag[src][i])["dpsi"]) for i in idx])
        figs["psi"][1][0].plot(GRID, np.rad2deg(np.median(ps, 0)), color=col, label=src.upper())
        figs["psi"][1][0].fill_between(GRID, np.rad2deg(np.percentile(ps, 25, 0)),
                                       np.rad2deg(np.percentile(ps, 75, 0)), color=col, alpha=0.15)
        vshape = []
        for i in idx:
            dsrc = zef[i]["d"] if src == "zef" else ag[src][i]["d"]
            tt = zef[i]["t"] if src == "zef" else ag[src][i]["t"]
            if len(dsrc) < 3: continue
            v = np.linalg.norm(np.diff(dsrc, axis=0), axis=1) / np.diff(tt).clip(min=1e-9)
            vshape.append(np.interp(GRID, np.linspace(0, 1, len(v)), v))
        V = np.stack(vshape)
        axv[0].plot(GRID, np.median(V, 0), color=col, label=src.upper())
        axv[0].fill_between(GRID, np.percentile(V, 25, 0), np.percentile(V, 75, 0), color=col, alpha=0.15)
        axv[1].hist(V.flatten(), bins=40, range=(0, 8), density=True, histtype="step", color=col, label=src.upper())
    figs["psi"][1][0].set_title("heading change (deg)"); figs["psi"][1][0].legend(fontsize=8)
    figs["psi"][1][0].grid(alpha=0.3); figs["psi"][1][1].axis("off")
    axv[0].set_title("|v| (BL/s) vs normalized time"); axv[1].set_title("speed distribution")
    axv[0].legend(fontsize=8); axv[1].legend(fontsize=8)
    figs["disp"][0].suptitle(hn); figs["psi"][0].suptitle(hn); figv.suptitle(hn)
    figs["disp"][0].tight_layout(); figs["disp"][0].savefig(OUT / "displacement_plots" / f"{hn}.png", dpi=120)
    figs["psi"][0].tight_layout(); figs["psi"][0].savefig(OUT / "heading_plots" / f"{hn}.png", dpi=120)
    figv.tight_layout(); figv.savefig(OUT / "velocity_plots" / f"{hn}.png", dpi=120)
    plt.close("all")

# chamfer distributions + fidelity vs target distance / duration
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
for m in ("pca", "cpg"):
    axs[0].hist([r["cd_path"] for r in rows[m]], bins=25, histtype="step", color=COL[m], label=m.upper())
    axs[1].scatter([r["tgt_disp"] for r in rows[m]], [r["cd_path"] for r in rows[m]],
                   s=14, color=COL[m], alpha=0.7, label=m.upper())
    axs[2].scatter([r["dur_z"] for r in rows[m]], [r["cd_path"] for r in rows[m]],
                   s=14, color=COL[m], alpha=0.7, label=m.upper())
axs[0].set_title("path-Chamfer distribution (BL)"); axs[0].legend()
axs[1].set_title("path-Chamfer vs target displacement (BL)"); axs[1].set_xlabel("target disp (BL)"); axs[1].legend()
axs[2].set_title("path-Chamfer vs ZeF segment duration (s)"); axs[2].set_xlabel("ZeF duration (s)"); axs[2].legend()
for a in axs: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "chamfer_distributions" / "chamfer.png", dpi=130)
plt.close(fig)

print(json.dumps({m: {"cd_path": metrics[m]["all"]["cd_path"]["median"],
                      "cd_time": metrics[m]["all"]["cd_time"]["median"],
                      "dx_err": metrics[m]["all"]["dx_err"]["median"],
                      "v_err": metrics[m]["all"]["v_err_norm"]["median"],
                      "psi_rmse": metrics[m]["all"]["psi_rmse"]["median"],
                      "success": metrics[m]["all"]["task_success_rate"]} for m in R}, indent=2))
print("wrote metrics.json + plots under", OUT)
