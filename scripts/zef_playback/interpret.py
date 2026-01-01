"""Per-PC functional interpretation from the sweep + combination envs (offline, after analyze.py).

For each single-mode sweep env (kappa = mean + 2*sigma_i*sin(2*pi*0.25Hz*t) * v_i):
  - SHAPE: the target + achieved kappa profile at the +2sigma and -2sigma extremes,
  - TURNING: correlation of the coefficient c(t) with the measured yaw RATE + net yaw drift,
  - PROPULSION: net forward displacement vs the coast env over the same window,
  - realizability: command clip fraction + tracking.

For the clip-combination envs the marginal effect of each added PC is the difference of their
speed/fidelity numbers (computed in analyze.py; re-read here to assemble one table).

Outputs: demo_out/zef_playback/interpretation.json + plots/interpret_pc*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curvature_utils as cu

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


OUT = Path(_P("${FISH_ROOT}/demo_out/zef_playback"))
PLOTS = OUT / "plots"


def main():
    r = np.load(OUT / "recording.npz")
    meta = json.loads((OUT / "meta.json").read_text())
    cache = np.load(OUT / "analysis_cache.npz")
    metrics = json.loads((OUT / "metrics.json").read_text())
    basis = np.load(_P("${FISH_ROOT}/demo_out/zef_manifold/pca_basis.npz"))
    play = cache["play_steps"]
    kap_sim = cache["kappa_sim"]
    dt = float(r["dt"])
    T = play.shape[0]
    sig = r["sigma"]
    sweep_hz = float(r["sweep_hz"])
    exps = meta["experiments"]
    out = {}

    coast_disp = metrics["experiments"]["coast"]["net_disp_m"]
    for e, exp in enumerate(exps):
        if exp["kind"] != "sweep":
            continue
        pc = exp["pc"]
        name = exp["name"]
        t = np.arange(T) * dt
        c = 2.0 * sig[pc] * np.sin(2 * np.pi * sweep_hz * t)      # commanded coefficient
        rs = r["root_state"][play, e]
        fwd = cu.quat_rotate_np(rs[:, 3:7], np.tile([1.0, 0, 0], (T, 1)))
        yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
        yaw_rate = np.gradient(yaw, dt)
        m = np.ones(T, bool)
        m[: int(1.0 / dt)] = False
        term = r["terminated"][play, e]
        if term.any():
            m[int(np.argmax(term)):] = False
        # correlations: instantaneous shape follows c with the PD lag; yaw RATE is the turning
        # signature (a turning mode holds a bent shape -> steady yaw rate while c != 0)
        r_yawrate = float(np.corrcoef(c[m], yaw_rate[m])[0, 1])
        # lateral drift of the head, in the body frame at sweep start
        mtr = metrics["experiments"][name]
        # extreme shapes
        i_hi = int(np.argmax(c * m)); i_lo = int(np.argmin(c + 1e9 * (~m)))
        kt = r["kappa_targets"][:, e]
        out[name] = {
            "pc": pc + 1, "sigma": float(sig[pc]),
            "corr_coeff_vs_yaw_rate": r_yawrate,
            "net_yaw_deg": mtr["net_yaw_deg"],
            "net_disp_m": mtr["net_disp_m"], "disp_vs_coast_m": mtr["net_disp_m"] - coast_disp,
            "v_fwd_bl": mtr["v_fwd_bl"], "tail_amp_bl": mtr["tail_amp_bl"],
            "clip_frac": mtr["clip_frac"],
            "fid_total_rmse": (mtr.get("fid_total") or {}).get("rmse"),
        }
        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].plot(cu.S_STATIONS, kt[i_hi], "k-", lw=2, label="target +2sig")
        axs[0].plot(cu.S_STATIONS, kap_sim[i_hi, e], "r-", label="sim +2sig")
        axs[0].plot(cu.S_STATIONS, kt[i_lo], "k--", lw=2, label="target -2sig")
        axs[0].plot(cu.S_STATIONS, kap_sim[i_lo, e], "r--", label="sim -2sig")
        axs[0].set_xlabel("s"); axs[0].set_ylabel("kappa*BL"); axs[0].legend(fontsize=8)
        axs[0].set_title(f"PC{pc+1} extreme shapes")
        axs[1].plot(t[m], c[m], label="coeff c(t)")
        axs[1].plot(t[m], yaw_rate[m] / max(np.abs(yaw_rate[m]).max(), 1e-6) * np.abs(c).max(),
                    label="yaw rate (scaled)")
        axs[1].legend(fontsize=8); axs[1].set_xlabel("t (s)")
        axs[1].set_title(f"corr(c, yaw rate) = {r_yawrate:+.2f}")
        axs[2].plot(rs[m, 0], rs[m, 1], "-", lw=1)
        axs[2].plot(rs[m][0, 0], rs[m][0, 1], "go", label="start")
        axs[2].plot(rs[m][-1, 0], rs[m][-1, 1], "rs", label="end")
        axs[2].set_aspect("equal"); axs[2].legend(fontsize=8)
        axs[2].set_title(f"root path (net yaw {mtr['net_yaw_deg']:+.0f} deg)")
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(PLOTS / f"interpret_{name}.png", dpi=130)
        plt.close(fig)

    # marginal effect of adding PCs (clip combos)
    ladder = ["pc1", "pc12_95", "pc123", "pc1234_99", "pc_all"]
    out["combo_ladder"] = {
        n: {k: metrics["experiments"][n].get(k) for k in
            ("v_fwd_bl", "tail_amp_bl", "net_yaw_deg", "tailbeat_hz", "clip_frac")}
        | {"fid_rmse": (metrics["experiments"][n].get("fid_total") or {}).get("rmse"),
           "fid_corr": (metrics["experiments"][n].get("fid_total") or {}).get("temporal_corr")}
        for n in ladder if n in metrics["experiments"]
    }
    # mapping / lag / speed ablations
    out["ablations"] = {
        n: {"v_fwd_bl": metrics["experiments"][n]["v_fwd_bl"],
            "fid_rmse": (metrics["experiments"][n].get("fid_total") or {}).get("rmse"),
            "fid_corr": (metrics["experiments"][n].get("fid_total") or {}).get("temporal_corr"),
            "clip_frac": metrics["experiments"][n]["clip_frac"]}
        for n in ("pc12_95", "pc12_lead", "pc12_half", "pc12_voronoi")
        if n in metrics["experiments"]
    }
    (OUT / "interpretation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {OUT / 'interpretation.json'} + plots/interpret_*.png")


if __name__ == "__main__":
    main()
