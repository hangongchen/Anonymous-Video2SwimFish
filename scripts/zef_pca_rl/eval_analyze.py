"""Offline analysis of eval_pca_swim.py recordings: metrics.json + plots + top-view videos.

Usage: <env python> scripts/zef_pca_rl/eval_analyze.py <eval_dir>   (dir with eval_recording.npz)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, _P("${FISH_ROOT}/scripts/zef_playback"))
import curvature_utils as cu

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


PANEL_NPZ = _P("${FISH_ROOT}/SimFishLib/fish_asset_pipeline/generated_usd_dataset/"
             "misty_minnow_fixed/panel_hydro.npz")
STEADY_SKIP_S = 2.0        # drop the first 2 s (gait spin-up) from steady metrics


def cond_names(n_policy):
    return ["frozen", "scripted_wave", "zef_playback"] + [f"policy_{i}" for i in range(n_policy)]


def main():
    d = Path(sys.argv[1])
    r = np.load(d / "eval_recording.npz")
    plots = d / "plots"; plots.mkdir(exist_ok=True)
    vids = d / "videos"; vids.mkdir(exist_ok=True)
    dt, BL = float(r["dt"]), float(r["body_length"])
    T, E = r["v_body"].shape[:2]
    names = cond_names(int(r["n_policy"]))
    geo = {"link_order": r["link_order"],
           "tips": {"nose": (int(r["tip_nose_bone"]), r["tip_nose_r"]),
                    "tail": (int(r["tip_tail_bone"]), r["tip_tail_r"])},
           "body_length": BL}
    m0 = int(STEADY_SKIP_S / dt)

    # measured curvature + midline per step/env (same estimator as the playback validation)
    kap = np.full((T, E, cu.K_STATIONS), np.nan, np.float32)
    mid = np.full((T, E, 100, 2), np.nan, np.float32)
    for t in range(T):
        for e in range(E):
            p = cu.kappa_from_bones(r["bone_pos"][t, e], r["bone_quat"][t, e], geo)
            if p is not None:
                kap[t, e], mid[t, e] = p["kappa_bl"], p["midline"]
    np.savez_compressed(d / "eval_kappa_cache.npz", kappa_meas=kap, midline=mid)

    metrics = {}
    tt = np.arange(T) * dt
    for e, name in enumerate(names):
        sl = slice(m0, T)
        term = r["terminated"][:, e]
        if term.any():
            sl = slice(m0, int(np.argmax(term)))
        v_b = r["v_body"][sl, e]
        v_fwd_bl = v_b[:, 0] / BL                               # body_forward_sign=+1
        v_lat_bl = np.sqrt(v_b[:, 1] ** 2 + v_b[:, 2] ** 2) / BL
        # yaw / heading from world heading vector
        q = r["root_state"][:, e, 3:7]
        fwd = cu.quat_rotate_np(q, np.tile([1.0, 0, 0], (T, 1)))
        yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
        net_yaw = float(np.rad2deg(yaw[sl][-1] - yaw[sl][0]))
        heading_err = float(np.rad2deg(np.abs(yaw[sl] - yaw[m0]).mean()))
        # tail-beat frequency from the tail-tip lateral offset
        ax = fwd[:, :2] / np.linalg.norm(fwd[:, :2], axis=1, keepdims=True).clip(1e-6)
        dvec = mid[:, e, -1] - mid[:, e, 0]
        tail_lat = dvec[:, 0] * (-ax[:, 1]) + dvec[:, 1] * ax[:, 0]
        sig = tail_lat[sl] - np.nanmean(tail_lat[sl])
        freq = None
        if np.isfinite(sig).all() and len(sig) > 64:
            f = np.fft.rfftfreq(len(sig), dt)
            p = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
            p[f < 0.2] = 0
            freq = float(f[int(np.argmax(p))])
        # curvature tracking (kappa-driven envs only: zef + policy)
        kc, km = r["kappa_cmd"][sl, e], kap[sl, e]
        ok = np.isfinite(km).all(1) & (np.abs(km).max(1) < 20)
        k_rmse = float(np.sqrt(((km[ok] - kc[ok]) ** 2).mean())) if ok.sum() > 10 else None
        # joint tracking + energy
        qa, qc = r["joint_pos"][sl, e], r["pos_targets"][sl, e]
        track = float(qa.std(0).sum() / max(qc.std(0).sum(), 1e-9))
        jrmse = float(np.sqrt(((qa - qc) ** 2).mean()))
        jv = r["joint_vel"][sl, e]
        energy = float((jv ** 2).mean())
        # PD mechanical power estimate: tau ~ k(q_tgt - q) - c qdot ; P = tau * qdot (positive part)
        k_stiff, c_damp = 120.0, 6.0
        tau = k_stiff * (qc - qa) - c_damp * jv
        power = float(np.clip(tau * jv, 0, None).sum(1).mean())
        metrics[name] = {
            "v_fwd_bl_mean": float(v_fwd_bl.mean()), "v_fwd_bl_last5s": float(v_fwd_bl[-int(5 / dt):].mean()),
            "v_lat_bl_rms": float(np.sqrt((v_lat_bl ** 2).mean())),
            "net_yaw_deg": net_yaw, "heading_err_deg_mean": heading_err,
            "tailbeat_hz": freq,
            "tail_amp_bl": float((np.nanpercentile(tail_lat[sl], 95) - np.nanpercentile(tail_lat[sl], 5)) / BL),
            "kappa_rmse": k_rmse, "joint_track_ratio": track, "joint_rmse_rad": jrmse,
            "energy_jv2": energy, "mech_power_w": power,
            "reward_mean": float(r["reward"][sl, e].mean()),
            "a1_rms": float(np.sqrt((r["a"][sl, e, 0] ** 2).mean())),
            "a2_rms": float(np.sqrt((r["a"][sl, e, 1] ** 2).mean())),
            "terminated": int(term.sum()),
            "net_disp_m": float(np.linalg.norm(r["root_state"][sl, e][-1, :2] - r["root_state"][sl, e][0, :2])),
        }

    # aggregate policy envs
    pol = [metrics[n] for n in names if n.startswith("policy_")]
    metrics["policy_mean"] = {k: float(np.mean([p[k] for p in pol if p[k] is not None]))
                              for k in pol[0] if isinstance(pol[0][k], (int, float)) and pol[0][k] is not None}
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("policy_") or k == "policy_mean"},
                     indent=2))

    # ---------------- plots ----------------
    e_pol = 3
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs[0, 0].plot(r["a"][:, e_pol, 0], r["a"][:, e_pol, 1], lw=0.6)
    axs[0, 0].plot(r["a"][:, 2, 0], r["a"][:, 2, 1], lw=0.6, alpha=0.6, label="ZeF playback")
    axs[0, 0].set_xlabel("a1"); axs[0, 0].set_ylabel("a2"); axs[0, 0].set_title("a1-a2 phase portrait (policy vs ZeF)")
    axs[0, 0].legend(fontsize=8)
    axs[0, 1].plot(tt, r["a"][:, e_pol, 0], label="a1")
    axs[0, 1].plot(tt, r["a"][:, e_pol, 1], label="a2")
    axs[0, 1].set_title("policy coefficient trajectories"); axs[0, 1].legend(fontsize=8)
    for e, nm in [(0, "frozen"), (1, "wave"), (2, "zef"), (e_pol, "policy")]:
        axs[1, 0].plot(tt, r["v_body"][:, e, 0] / BL, label=nm, lw=1)
    axs[1, 0].set_title("body-frame forward velocity (BL/s)"); axs[1, 0].legend(fontsize=8)
    axs[1, 0].set_xlabel("t (s)")
    for e, nm in [(1, "wave"), (2, "zef"), (e_pol, "policy")]:
        q = r["root_state"][:, e, 3:7]
        fwd = cu.quat_rotate_np(q, np.tile([1.0, 0, 0], (T, 1)))
        axs[1, 1].plot(tt, np.rad2deg(np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))), label=nm, lw=1)
    axs[1, 1].set_title("yaw (deg)"); axs[1, 1].legend(fontsize=8); axs[1, 1].set_xlabel("t (s)")
    fig.tight_layout(); fig.savefig(plots / "eval_summary.png", dpi=130); plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    vm = 4.0
    axs[0].imshow(r["kappa_cmd"][:, e_pol].T, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm,
                  extent=[0, T * dt, 1, 0]); axs[0].set_title("policy commanded kappa_bl")
    axs[1].imshow(kap[:, e_pol].T, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm,
                  extent=[0, T * dt, 1, 0]); axs[1].set_title("measured kappa_bl")
    axs[1].set_xlabel("t (s)"); [a.set_ylabel("s") for a in axs]
    fig.tight_layout(); fig.savefig(plots / "kappa_heatmap_policy.png", dpi=130); plt.close(fig)

    # ---------------- videos (top view, one per condition) ----------------
    pd = np.load(PANEL_NPZ)
    rng = np.random.default_rng(0)
    sel = rng.choice(pd["panel_bone"].shape[0], 600, replace=False)
    pbi, prl = pd["panel_bone"][sel].astype(int), pd["panel_r_local"][sel].astype(np.float64)
    for e, nm in [(0, "frozen"), (1, "scripted_wave"), (2, "zef_playback"), (e_pol, "policy")]:
        bp, bq = r["bone_pos"][:, e], r["bone_quat"][:, e]
        pw = bp[:, pbi] + cu.quat_rotate_np(bq[:, pbi], prl[None].repeat(T, 0))
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        sc = ax.scatter(pw[0, :, 0], pw[0, :, 1], s=2, c="#7fb2d9", alpha=0.6)
        (ml,) = ax.plot(mid[0, e, :, 0], mid[0, e, :, 1], "-", color="crimson", lw=1.5)
        (trace,) = ax.plot([], [], "-", color="gray", lw=0.8)
        ax.set_aspect("equal"); ax.set_title(nm)
        txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)
        head = mid[:, e, 0]
        w = FFMpegWriter(fps=30, bitrate=2000)
        with w.saving(fig, str(vids / f"{nm}.mp4"), dpi=100):
            for t in range(0, T, 1):
                sc.set_offsets(pw[t, :, :2])
                ml.set_data(mid[t, e, :, 0], mid[t, e, :, 1])
                trace.set_data(head[:t + 1, 0], head[:t + 1, 1])
                cx, cy = np.nanmean(pw[t, :, 0]), np.nanmean(pw[t, :, 1])
                ax.set_xlim(cx - 0.6, cx + 0.6); ax.set_ylim(cy - 0.6, cy + 0.6)
                txt.set_text(f"t={t*dt:5.2f}s  v_fwd={r['v_body'][t,e,0]/BL:+.2f} BL/s")
                w.grab_frame()
        plt.close(fig)
        print(f"wrote {vids / (nm + '.mp4')}")
    print(f"\nAll outputs in {d}")


if __name__ == "__main__":
    main()
