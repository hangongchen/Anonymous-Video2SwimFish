"""Offline video rendering of the ZeF-mode playback (NO Isaac, no Replicator -- the known
FEM+camera stall is avoided entirely by drawing from the recorded states).

One top-view mp4 per experiment env, play phase only:
  LEFT   world-frame top view: body surface (panel cloud pinned to the recorded bone poses),
         bone chain, spline midline, path trace of the head.
  RIGHT  target vs achieved curvature profile kappa_bl(s) at the current frame, plus the
         per-frame command coefficients / joint targets.

Run AFTER analyze.py (reuses its kappa/midline cache):
  ${FISH_PYTHON} scripts/zef_playback/render_videos.py
Outputs: demo_out/zef_playback/videos/<name>.mp4
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import curvature_utils as cu

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


OUT = Path(_P("${FISH_ROOT}/demo_out/zef_playback"))
VID = OUT / "videos"
PANEL_NPZ = _P("${FISH_ROOT}/SimFishLib/fish_asset_pipeline/generated_usd_dataset/"
               "misty_minnow_fixed/panel_hydro.npz")
SMOKE = "--smoke" in sys.argv
ONLY = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
FPS = 30
STRIDE = 1            # render every control step (30 fps real time)
N_PANELS = 700        # subsampled surface points


def main():
    tag = "recording_smoke" if SMOKE else "recording"
    r = np.load(OUT / f"{tag}.npz")
    meta = json.loads((OUT / ("meta_smoke.json" if SMOKE else "meta.json")).read_text())
    cache = np.load(OUT / ("analysis_cache_smoke.npz" if SMOKE else "analysis_cache.npz"))
    VID.mkdir(parents=True, exist_ok=True)

    pd = np.load(PANEL_NPZ)
    rng = np.random.default_rng(0)
    sel = rng.choice(pd["panel_bone"].shape[0], N_PANELS, replace=False)
    pbi = pd["panel_bone"][sel].astype(int)
    prl = pd["panel_r_local"][sel].astype(np.float64)

    play = cache["play_steps"]
    kap_sim = cache["kappa_sim"]
    mid = cache["midline"]
    dt = float(r["dt"])
    exps = meta["experiments"]
    T = play.shape[0]
    sig = r["sigma"]

    for e, exp in enumerate(exps):
        name = exp["name"]
        if ONLY and name not in ONLY:
            continue
        bp = r["bone_pos"][play, e]                       # (T,B,3)
        bq = r["bone_quat"][play, e]
        kt = r["kappa_targets"][:, e]
        # precompute panel cloud world XY for all frames (vectorized)
        pw = bp[:, pbi] + cu.quat_rotate_np(bq[:, pbi], prl[None, :, :].repeat(T, 0))
        head = mid[:, e, 0]

        fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
        axL = fig.add_axes([0.05, 0.08, 0.55, 0.84])
        axR = fig.add_axes([0.68, 0.42, 0.29, 0.50])
        axC = fig.add_axes([0.68, 0.08, 0.29, 0.24])
        cx, cy = np.nanmean(pw[0, :, 0]), np.nanmean(pw[0, :, 1])
        half = 0.55
        scat = axL.scatter(pw[0, :, 0], pw[0, :, 1], s=2, c="#7fb2d9", alpha=0.6)
        (chain,) = axL.plot(bp[0, r["link_order"], 0], bp[0, r["link_order"], 1], "o-",
                            color="#1a4a72", ms=3, lw=1)
        (midl,) = axL.plot(mid[0, e, :, 0], mid[0, e, :, 1], "-", color="crimson", lw=1.5)
        (trace,) = axL.plot([], [], "-", color="gray", lw=0.8, alpha=0.7)
        axL.set_aspect("equal")
        axL.set_title(f"{name} -- top view (world XY)")
        title = axL.text(0.02, 0.98, "", transform=axL.transAxes, va="top", fontsize=9)

        s20 = cu.S_STATIONS
        (l_t,) = axR.plot(s20, kt[0], "-", color="k", lw=2, label="target")
        (l_s,) = axR.plot(s20, kap_sim[0, e], "-", color="crimson", lw=1.5, label="sim")
        vm = max(np.nanmax(np.abs(kt)) * 1.3, 2.0)
        axR.set_ylim(-vm, vm); axR.set_xlabel("s (head=0)"); axR.set_ylabel("kappa*BL")
        axR.legend(loc="upper right", fontsize=8)
        axR.set_title("curvature profile", fontsize=10)
        axR.grid(alpha=0.3)

        fam1 = r["fam1"]
        q_cmd = r["A_play"][:, e][:, fam1] * float(r["pos_action_scale"])
        tt = np.arange(T) * dt
        axC.plot(tt, np.rad2deg(q_cmd), lw=0.7)
        cursor = axC.axvline(0, color="k", lw=1)
        axC.set_xlabel("t (s)"); axC.set_ylabel("joint tgt (deg)")
        axC.set_title("commanded joint targets", fontsize=9)

        path = VID / f"{name}.mp4"
        writer = FFMpegWriter(fps=FPS, bitrate=2500)
        with writer.saving(fig, str(path), dpi=100):
            for t in range(0, T, STRIDE):
                scat.set_offsets(pw[t, :, :2])
                lo = r["link_order"]
                chain.set_data(bp[t, lo, 0], bp[t, lo, 1])
                midl.set_data(mid[t, e, :, 0], mid[t, e, :, 1])
                trace.set_data(head[: t + 1, 0], head[: t + 1, 1])
                cx, cy = np.nanmean(pw[t, :, 0]), np.nanmean(pw[t, :, 1])
                axL.set_xlim(cx - half, cx + half); axL.set_ylim(cy - half, cy + half)
                l_t.set_ydata(kt[t]); l_s.set_ydata(kap_sim[t, e])
                cursor.set_xdata([t * dt])
                title.set_text(f"t={t*dt:5.2f}s")
                writer.grab_frame()
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
