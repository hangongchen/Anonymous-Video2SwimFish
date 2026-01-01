#!/usr/bin/env python
"""Build a CLEAN AMP reference from ZebraFish-05 (the sequence the fish ASSET was carved from).

Why this replaces demo_out/zef_amp_ref/amp_reference.npz (ZebraFish-01):
  - ZebraFish-01 has 2 fish; 44% of frames are occluded -> the carved bend feature was white noise,
    the discriminator separated it from ITSELF (AUC 0.99), and r_style pinned flat. (See the audit.)
  - ZebraFish-05 is a SINGLE fish, 0% occlusion, coherent ~2-3 Hz tail beat, and is the SAME
    individual the asset morphology came from -> no cross-fish retarget.

Extraction (fixes the second defect -- the 3D two-view carve was lossy for the bend envelope):
  1. Top-view silhouette mask per frame (lateral undulation is fully in the top view).
  2. Body axis = smoothed head-velocity heading (Di Santo: align to mean motion direction). This is
     STABLE against the tail beat and follows turns, so it does not rotate the bend out (the carve's
     per-frame rotate_upright+PCA put the oscillation node at mid-body -> flat envelope).
  3. Project mask pixels into (along-body, lateral), run the IDENTICAL spine_bend_profile the sim env
     uses in _bend_feature -> real and sim features share one transform (AMP requires this).

Output: demo_out/zef05_amp_ref/amp_reference.npz  {feat(T,K), frame, segment_id, is_turn, K}
Prints progress_saturation_speed = median positive forward speed in BL/s (for the reward cfg).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_amp_features import anterior_aligned_profile

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


BL_CM = 3.4   # adult zebrafish body length (cm), for BL/s speed normalization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-root", default=_P("${FISH_ROOT}/demo_out/zef05_seg/ZebraFish-05/imgT/mask"))
    ap.add_argument("--gt", default=_P("${ZEF_ROOT}/gt/gt.txt"))
    ap.add_argument("--out", default=_P("${FISH_ROOT}/demo_out/zef05_amp_ref/amp_reference.npz"))
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--src-fps", type=float, default=60.0)
    ap.add_argument("--stride", type=int, default=2, help="60fps -> 30fps to match 30 Hz control")
    ap.add_argument("--smooth", type=int, default=5, help="temporal smoothing taps at src fps (odd)")
    ap.add_argument("--turn-thresh", type=float, default=1.0, help="|yaw_rate| rad/s above = turning")
    args = ap.parse_args()

    gt = np.loadtxt(args.gt, delimiter=",")
    gt = gt[gt[:, 1] == 1]
    fr_all = gt[:, 0].astype(int)
    pos3d = gt[:, 2:5]                                   # cm, 3D head
    headpx = {int(r[0]): np.array([r[5], r[6]]) for r in gt}   # camT head pixel

    # smoothed head-velocity heading (image px) -> stable body axis per frame
    def smooth(x, w=9):
        k = np.ones(w) / w
        return np.stack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])], 1)
    hp = np.array([headpx[f] for f in fr_all])
    hp_s = smooth(hp, 9)
    hvel = np.gradient(hp_s, axis=0)
    heading_px = {int(fr_all[i]): (hvel[i] / (np.linalg.norm(hvel[i]) + 1e-9)) for i in range(len(fr_all))}

    files = {int(os.path.basename(f)[:-4]): f for f in glob.glob(args.mask_root + "/*.png")}
    # extract at FULL src fps, temporally smooth (removes per-frame mask/axis jitter without touching
    # the ~3 Hz beat), THEN downsample to the control rate -- 30 fps extraction alone is too jittery.
    frames_full = sorted(f for f in files if f in headpx)

    profs, kept = [], []
    for fr in frames_full:
        m = imageio.imread(files[fr])
        if m.ndim == 3:
            m = m[..., 0]
        ys, xs = np.nonzero(m > 127)
        if len(xs) < 50:
            continue
        P = np.stack([xs, ys], 1).astype(np.float64)
        # initial axis = head->centroid, ordering only; anterior_aligned_profile re-fits the axis to the
        # ANTERIOR body tangent so sharply-bent (turning) frames are not mis-projected (Problem 1 fix).
        ctr = P.mean(0)
        axis = ctr - headpx[fr]
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        lat = np.array([-axis[1], axis[0]])
        rel = P - headpx[fr]                             # origin at head
        along = rel @ axis
        side = rel @ lat
        profs.append(anterior_aligned_profile(along, side, args.K))   # SAME fn as sim _amp_feature
        kept.append(fr)

    feat_full = np.stack(profs).astype(np.float32)             # (T_full, K) at src fps
    kept_full = np.array(kept, dtype=np.int64)
    # temporal smooth (5-tap @60fps ~= 6 Hz cutoff, above the ~3 Hz beat) then downsample to control fps
    w = args.smooth
    k = np.ones(w) / w
    feat_sm = np.stack([np.convolve(feat_full[:, j], k, mode="same") for j in range(args.K)], 1)
    feat = feat_sm[:: args.stride].astype(np.float32)
    kept = kept_full[:: args.stride]
    seg = np.zeros(len(kept), dtype=np.int64)            # one contiguous block

    # trajectory + VELOCITY CHANNELS from 3D head (cm) at the strided rate. Heading = smoothed
    # horizontal velocity direction; decompose velocity into forward/lateral/vertical (BL/s) + yaw.
    idx = np.searchsorted(fr_all, kept)
    p = pos3d[idx]
    dt = args.stride / args.src_fps
    psm = np.stack([np.convolve(p[:, i], np.ones(5) / 5, mode="same") for i in range(3)], 1)
    vel = np.gradient(psm, axis=0) / dt                  # cm/s
    speed_bl = np.linalg.norm(vel, axis=1) / BL_CM
    hd = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))     # horizontal heading
    yaw = np.gradient(hd) / dt                           # rad/s
    is_turn = np.abs(yaw) > args.turn_thresh
    hdir = np.stack([np.cos(hd), np.sin(hd)], 1)         # unit heading (horizontal)
    v_fwd = np.sum(vel[:, :2] * hdir, axis=1) / BL_CM    # forward (BL/s), +=head-first
    # VELOCITY CHANNELS = [v_fwd, yaw] only. Excluded: v_lat (0 by construction when heading=velocity
    # dir, and body-axis-relative lateral conflates natural recoil with crab -> risks penalizing recoil)
    # and v_vert (ref std ~0.9 BL/s but the sim is z-band-contained -> the disc would demand vertical
    # motion the containment forbids). v_fwd directly prevents the FROZEN pose (frozen -> v_fwd=0 !=
    # reference), yaw carries the turn distribution (73% of frames). Both unambiguous on sim + ref.
    vel_ch = np.stack([v_fwd, yaw], 1).astype(np.float32)                  # (T, 2)
    sat = float(np.median(v_fwd[v_fwd > 0.05]))

    # Phi per-frame feature = [bend(K), v_fwd, yaw]  -> (T, K+2)
    feat_full_out = np.concatenate([feat, vel_ch], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, feat=feat_full_out, frame=kept, segment_id=seg, is_turn=is_turn, K=args.K,
             n_vel=2, speed_bl=speed_bl.astype(np.float32), yaw_rate=yaw.astype(np.float32))
    print(f"[build_zef05_reference] {len(kept)} frames @ {args.src_fps/args.stride:.0f} fps "
          f"-> feat {feat_full_out.shape} (K={args.K} bend + 2 vel [v_fwd,yaw], {int(is_turn.sum())} turn)")
    print(f"  velocity channels (raw): v_fwd med %.2f std %.2f | yaw std %.2f"
          % (np.median(v_fwd), v_fwd.std(), yaw.std()))
    print(f"  wrote {args.out}")
    print(f"  reference speed: median {np.median(speed_bl):.2f} BL/s, p90 {np.percentile(speed_bl,90):.2f}")
    print(f"  >>> progress_saturation_speed (median positive forward speed) = {sat:.2f} BL/s")


if __name__ == "__main__":
    main()
