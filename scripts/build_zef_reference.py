#!/usr/bin/env python
"""Step 2a of the AMP pipeline: select a REFERENCE frame set (mix of turn-rich + straight bouts)
from a 3D-ZeF track and export the per-frame trajectory features (speed, yaw-rate) that will be
paired with the carved body-shape clouds to form the AMP motion dataset.

Outputs (under --out):
  frames.txt        one frame number per line (the frames to segment + carve)
  traj_feats.npz    frame, speed(m/s), yaw_rate(rad/s, signed), is_turn(bool)  -- aligned to frames.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from zef_turn_analysis import track, kinematics  # same dir on sys.path when run from scripts/

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)



def pick_windows(is_turn, frames, win, n_turn, n_straight):
    dens = np.array([is_turn[i:i + win].mean() for i in range(0, len(frames) - win, win)])
    starts = np.arange(len(dens)) * win
    order_turn = np.argsort(dens)[::-1]
    order_straight = np.argsort(dens)
    chosen = []
    for idx in order_turn:
        if len(chosen) >= n_turn:
            break
        chosen.append(starts[idx])
    ns = 0
    for idx in order_straight:
        if ns >= n_straight:
            break
        if starts[idx] not in chosen:
            chosen.append(starts[idx]); ns += 1
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=_P("${ZEF_ROOT}/data/ZebraFish-01/gt/gt.txt"))
    ap.add_argument("--fish-id", type=int, default=1)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--smooth-win", type=int, default=15)
    ap.add_argument("--turn-thresh", type=float, default=1.0)
    ap.add_argument("--min-speed", type=float, default=0.03)
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--n-turn-windows", type=int, default=3)
    ap.add_argument("--n-straight-windows", type=int, default=2)
    ap.add_argument("--frame-stride", type=int, default=2, help="carve every Nth selected frame (60->30fps)")
    ap.add_argument("--out", default=_P("${FISH_ROOT}/demo_out/zef_amp_ref"))
    args = ap.parse_args()

    frames, pos = track(Path(args.gt), args.fish_id)
    speed, yaw, _ = kinematics(frames, pos, args.fps, args.smooth_win)
    moving = speed > args.min_speed
    is_turn = (np.abs(yaw) > args.turn_thresh) & moving

    starts = pick_windows(is_turn, frames, args.window, args.n_turn_windows, args.n_straight_windows)
    sel = []
    for s in starts:
        sel.extend(range(s, min(s + args.window, len(frames)), args.frame_stride))
    sel = sorted(set(sel))
    sel_frames = frames[sel]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "frames.txt", sel_frames, fmt="%d")
    np.savez(out / "traj_feats.npz",
             frame=sel_frames, speed=speed[sel].astype(np.float32),
             yaw_rate=yaw[sel].astype(np.float32), is_turn=is_turn[sel])
    print(f"[build_zef_reference] selected {len(sel_frames)} frames from "
          f"{args.n_turn_windows} turn + {args.n_straight_windows} straight windows "
          f"(starts@frame {[int(frames[s]) for s in starts]})")
    print(f"  turn frames in selection: {int(is_turn[sel].sum())}/{len(sel)}  "
          f"speed {speed[sel][moving[sel]].mean():.3f} m/s  |yaw| {np.abs(yaw[sel]).mean():.2f} rad/s")
    print(f"  wrote {out/'frames.txt'} , {out/'traj_feats.npz'}")


if __name__ == "__main__":
    main()
