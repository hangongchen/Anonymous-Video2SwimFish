"""Offline analysis of the ZeF-mode playback recording (NO Isaac needed).

Reads demo_out/zef_playback/recording.npz, reconstructs the simulated centerline curvature at
every play step with the SAME estimator used for calibration, and reports:

  CURVATURE FIDELITY (primary): RMSE, max error, temporal correlation vs each experiment's own
  target kappa_bl(s,t) -- split into REPRESENTATION error (kappa_ideal = rest + Phi q_cmd, i.e.
  perfect joints through the calibrated linear map, including command clipping) and DYNAMIC
  error (sim vs ideal), so "7 hinges + rigid tips can't draw it" is separated from "the PD/FEM
  didn't track it".

  PHYSICAL EVALUATION: forward speed (m/s, BL/s), lateral speed rms, tail-beat amplitude +
  frequency, per-joint tracking ratio + phase lag, stability (max jvel, terminations, z/pitch
  drift), net displacement vs the zero-action coast env, net yaw (turning).

Outputs (all under demo_out/zef_playback/):
  analysis_cache.npz   kappa_sim + midlines per play step (reused by render_videos.py)
  metrics.json         every number, per experiment
  plots/*.png          Phi heatmap, per-env kappa(s,t) target|sim|error heatmaps, fidelity bars,
                       speed/tail bars, sweep-mode interpretation panels
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
SMOKE = "--smoke" in sys.argv
TAG = "recording_smoke" if SMOKE else "recording"
TRANSIENT_S = 1.0          # ignore the first second of play (spin-up)
WRAP_MASK_S = 0.5          # ignore +/- this around the clip loop point


def load():
    r = np.load(OUT / f"{TAG}.npz", allow_pickle=False)
    meta = json.loads((OUT / ("meta_smoke.json" if SMOKE else "meta.json")).read_text())
    geo = {
        "link_order": r["link_order"],
        "tips": {"nose": (int(r["tip_nose_bone"]), r["tip_nose_r"]),
                 "tail": (int(r["tip_tail_bone"]), r["tip_tail_r"])},
        "body_length": float(r["body_length"]),
    }
    return r, meta, geo


def quat_rotate(q, v):
    return cu.quat_rotate_np(q, v)


def build_cache(r, meta, geo):
    """kappa_sim (T,E,20) + midline (T,E,100,2) + head/tail world points for the PLAY phase."""
    cache_p = OUT / ("analysis_cache_smoke.npz" if SMOKE else "analysis_cache.npz")
    play = np.where(r["phase_of_step"] == "play")[0]
    rec_mtime = (OUT / f"{TAG}.npz").stat().st_mtime
    if cache_p.exists() and cache_p.stat().st_mtime > rec_mtime:
        c = np.load(cache_p)
        if c["play_steps"].shape[0] == play.shape[0]:
            print(f"[cache] reusing {cache_p}")
            return dict(c)
    T, E = play.shape[0], r["bone_pos"].shape[1]
    kap = np.full((T, E, cu.K_STATIONS), np.nan, np.float32)
    mid = np.full((T, E, 100, 2), np.nan, np.float32)
    for ti, t in enumerate(play):
        for e in range(E):
            p = cu.kappa_from_bones(r["bone_pos"][t, e], r["bone_quat"][t, e], geo)
            if p is not None:
                kap[ti, e] = p["kappa_bl"]
                mid[ti, e] = p["midline"]
        if ti % 100 == 0:
            print(f"[cache] {ti}/{T}")
    out = {"kappa_sim": kap, "midline": mid, "play_steps": play}
    np.savez_compressed(cache_p, **out)
    print(f"[cache] wrote {cache_p}")
    return out


def valid_mask(r, e, play, dt, exp=None):
    """Per-play-step validity: drop spin-up, the clip wrap, and anything after a termination.

    The wrap discontinuity exists ONLY for full-speed clip envs (the 253-step clip looped
    twice); sweeps/coast are continuous and the half-speed env plays the 505-frame clip once
    (its only jump is the final tiled frame). Applying wrap_step to every env wrongly dropped
    ~6% of valid frames from 6 of 13 envs (verification finding)."""
    T = play.shape[0]
    m = np.ones(T, bool)
    m[: int(TRANSIENT_S / dt)] = False
    is_clip_full = exp is not None and exp.get("kind") == "clip" and not exp.get("half_speed")
    w = int(r["wrap_step"])
    if is_clip_full and 0 < w < T:
        k = int(WRAP_MASK_S / dt)
        m[max(0, w - k): w + k] = False
    if exp is not None and exp.get("half_speed"):
        m[505:] = False                                   # tiled hold/jump after the single pass
    term = r["terminated"][play, e]
    if term.any():
        m[int(np.argmax(term)):] = False
    return m


def fidelity(kt, ks, m, sane=20.0):
    """RMSE / max err / temporal correlation between target (T,20) and sim (T,20) on mask m.

    Frames whose measured |kappa_bl| exceeds `sane` are DEGENERATE SPLINE FITS (a cusp/fold in
    the 10-point centerline -- physically the chain cannot exceed ~10), not physics; they are
    excluded and counted, otherwise a handful of frames dominates the RMSE."""
    a, b = kt[m], ks[m]
    ok = np.isfinite(b).all(1) & (np.abs(b).max(1) < sane)
    n_outlier = int((np.isfinite(b).all(1) & ~(np.abs(b).max(1) < sane)).sum())
    a, b = a[ok], b[ok]
    if len(a) < 10:
        return {"rmse": None, "max_err": None, "temporal_corr": None, "n": int(len(a))}
    err = b - a
    per_station_r = []
    for s in range(a.shape[1]):
        if a[:, s].std() > 1e-6 and b[:, s].std() > 1e-6:
            per_station_r.append(float(np.corrcoef(a[:, s], b[:, s])[0, 1]))
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "max_err": float(np.abs(err).max()),
        "temporal_corr": float(np.mean(per_station_r)) if per_station_r else None,
        "per_station_corr": per_station_r,
        "n": int(len(a)),
        "outlier_frames": n_outlier,
    }


def main():
    r, meta, geo = load()
    cache = build_cache(r, meta, geo)
    PLOTS.mkdir(parents=True, exist_ok=True)
    dt = float(r["dt"])
    BL = geo["body_length"]
    play = cache["play_steps"]
    T, E = play.shape[0], r["bone_pos"].shape[1]
    exps = meta["experiments"]
    fam1 = r["fam1"]
    scale = float(r["pos_action_scale"])
    Phi, k_rest = r["Phi"], r["kappa_rest"]
    kap_sim = cache["kappa_sim"]
    mid = cache["midline"]
    metrics = {"experiments": {}}

    # ---- Phi heatmap ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(Phi, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(Phi).max(), vmax=np.abs(Phi).max())
    ax.set_xlabel("joint (head -> tail)"); ax.set_ylabel("station s (head=0)")
    ax.set_title("Calibrated curvature influence Phi (kappa_bl per rad)")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(PLOTS / "phi_heatmap.png", dpi=130); plt.close(fig)

    coast_disp = None
    for e, exp in enumerate(exps):
        name = exp["name"]
        m = valid_mask(r, e, play, dt, exp)
        kt = r["kappa_targets"][:, e]                              # (T,20) this env's own target
        ks = kap_sim[:, e]
        q_cmd = r["A_play"][:, e][:, fam1] * scale                 # POST-clip commanded angles
        k_ideal = k_rest[None, :] + q_cmd @ Phi.T                  # linearized perfect-joint kappa

        res = {"kind": exp["kind"], "clip_frac": float(r["clip_frac"][e])}
        if exp["kind"] != "coast":
            res["fid_total"] = fidelity(kt, ks, m)                 # sim vs target (headline)
            res["fid_representation"] = fidelity(kt, k_ideal, m)   # map+hinges+clip ceiling
            res["fid_dynamic"] = fidelity(k_ideal, ks, m)          # tracking loss on top

        # ---- physical metrics (steady mask) ----
        rs = r["root_state"][play, e]                              # (T,13)
        fwd_w = quat_rotate(rs[:, 3:7], np.tile([1.0, 0, 0], (T, 1)))
        fwd_w /= np.linalg.norm(fwd_w, axis=1, keepdims=True).clip(1e-6)
        v = rs[:, 7:10]
        v_fwd = (v * fwd_w).sum(1)
        lat_w = np.stack([-fwd_w[:, 1], fwd_w[:, 0], np.zeros(T)], 1)
        lat_w /= np.linalg.norm(lat_w, axis=1, keepdims=True).clip(1e-6)
        v_lat = (v * lat_w).sum(1)
        head_xy, tail_xy = mid[:, e, 0], mid[:, e, -1]
        ax_xy = fwd_w[:, :2] / np.linalg.norm(fwd_w[:, :2], axis=1, keepdims=True).clip(1e-6)
        d = tail_xy - head_xy
        tail_lat = d[:, 0] * (-ax_xy[:, 1]) + d[:, 1] * ax_xy[:, 0]
        # tail-beat frequency from the detrended lateral tail signal
        sig = tail_lat[m] - np.nanmean(tail_lat[m])
        freq, freq_peaks = None, None
        if m.sum() > 32 and np.isfinite(sig).all():
            f = np.fft.rfftfreq(len(sig), dt)
            p = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
            p[f < 0.15] = 0.0                                      # kill DC/drift
            freq = float(f[int(np.argmax(p))])
            top = np.argsort(p)[::-1][:3]
            freq_peaks = [[round(float(f[i]), 2), round(float(p[i] / p.max()), 2)] for i in top]
        # per-joint tracking (amplitude ratio + lag) on the play window
        q_ach = r["joint_pos"][play, e][:, fam1]
        track_ratio, lag_ms = [], []
        for j in range(q_cmd.shape[1]):
            c_, a_ = q_cmd[m, j], q_ach[m, j]
            if c_.std() > 1e-4:
                track_ratio.append(float(a_.std() / c_.std()))
                xc = np.correlate(a_ - a_.mean(), c_ - c_.mean(), "full")
                lag_ms.append(float((np.argmax(xc) - (len(c_) - 1)) * dt * 1e3))
        # displacement, yaw, stability
        p0, p1 = rs[m][0, :3], rs[m][-1, :3]
        disp = float(np.linalg.norm((p1 - p0)[:2]))
        yaw = np.unwrap(np.arctan2(fwd_w[:, 1], fwd_w[:, 0]))
        net_yaw = float(np.rad2deg(yaw[m][-1] - yaw[m][0]))
        res.update({
            "v_fwd_mean": float(np.mean(v_fwd[m])), "v_fwd_bl": float(np.mean(v_fwd[m]) / BL),
            "v_lat_rms": float(np.sqrt(np.mean(v_lat[m] ** 2))),
            "tail_amp_m": float(np.percentile(tail_lat[m], 95) - np.percentile(tail_lat[m], 5)),
            "tail_amp_bl": float((np.percentile(tail_lat[m], 95) - np.percentile(tail_lat[m], 5)) / BL),
            "tailbeat_hz": freq, "tailbeat_peaks": freq_peaks,
            "track_ratio_mean": float(np.mean(track_ratio)) if track_ratio else None,
            "lag_ms_mean": float(np.mean(lag_ms)) if lag_ms else None,
            "net_disp_m": disp, "net_yaw_deg": net_yaw,
            "z_drift_m": float(rs[m][-1, 2] - rs[m][0, 2]),
            "jvel_max": float(r["joint_vel_max"][play, e][m].max()),
            "terminations": int(r["terminated"][play, e].sum()),
            "valid_frac": float(m.mean()),
        })
        if name == "coast":
            coast_disp = disp
        metrics["experiments"][name] = res

        # ---- per-env kappa(s,t) heatmap: target | sim | error ----
        if exp["kind"] != "coast":
            fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
            vm = max(np.nanmax(np.abs(kt)), 1.0)
            tt = np.arange(T) * dt
            for a_, dat, ttl in ((axs[0], kt, "target kappa_bl"),
                                 (axs[1], ks, "sim kappa_bl"),
                                 (axs[2], ks - kt, "error")):
                imh = a_.imshow(dat.T, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm,
                                extent=[tt[0], tt[-1], 1, 0])
                a_.set_ylabel("s"); a_.set_title(ttl, fontsize=9)
                fig.colorbar(imh, ax=a_)
            axs[2].set_xlabel("t (s)")
            fig.suptitle(f"{name}")
            fig.tight_layout()
            fig.savefig(PLOTS / f"kappa_st_{name}.png", dpi=120); plt.close(fig)

    # thrust vs coast
    for name, res in metrics["experiments"].items():
        if coast_disp is not None and name != "coast":
            res["disp_vs_coast_m"] = float(res["net_disp_m"] - coast_disp)

    # ---- summary bar charts -----------------------------------------------------------
    names = [x["name"] for x in exps if x["kind"] != "coast"]
    def _get(nm, *ks_):
        v = metrics["experiments"][nm]
        for k in ks_:
            v = v[k] if v is not None and v.get(k) is not None else None
            if v is None:
                return np.nan
        return v
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(names))
    axs[0].bar(x - 0.2, [_get(n, "fid_total", "rmse") for n in names], 0.2, label="total")
    axs[0].bar(x, [_get(n, "fid_representation", "rmse") for n in names], 0.2, label="representation")
    axs[0].bar(x + 0.2, [_get(n, "fid_dynamic", "rmse") for n in names], 0.2, label="dynamic")
    axs[0].set_xticks(x, names, rotation=45, ha="right"); axs[0].legend()
    axs[0].set_ylabel("kappa_bl RMSE"); axs[0].set_title("curvature fidelity")
    axs[1].bar(x, [_get(n, "fid_total", "temporal_corr") for n in names])
    axs[1].set_xticks(x, names, rotation=45, ha="right"); axs[1].set_ylim(0, 1)
    axs[1].set_title("temporal correlation (mean over stations)")
    fig.tight_layout(); fig.savefig(PLOTS / "fidelity_bars.png", dpi=130); plt.close(fig)

    all_names = [x["name"] for x in exps]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(all_names))
    axs[0].bar(x, [metrics["experiments"][n]["v_fwd_bl"] for n in all_names])
    axs[0].set_title("forward speed (BL/s)")
    axs[1].bar(x, [metrics["experiments"][n]["tail_amp_bl"] for n in all_names])
    axs[1].set_title("tail amplitude (BL)")
    axs[2].bar(x, [metrics["experiments"][n]["net_yaw_deg"] for n in all_names])
    axs[2].set_title("net yaw (deg)")
    for a_ in axs:
        a_.set_xticks(x, all_names, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(PLOTS / "physical_bars.png", dpi=130); plt.close(fig)

    (OUT / ("metrics_smoke.json" if SMOKE else "metrics.json")).write_text(
        json.dumps(metrics, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                      for k, v in metrics["experiments"].items()}, indent=2))
    print(f"\nwrote {OUT / ('metrics_smoke.json' if SMOKE else 'metrics.json')}")
    print(f"plots in {PLOTS}/")


if __name__ == "__main__":
    main()
