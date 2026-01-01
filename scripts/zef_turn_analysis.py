#!/usr/bin/env python
"""Characterize turning in a 3D-ZeF track: per-frame speed + yaw-rate, turn vs straight, and the
turn-rich frame windows. Feeds step 2 of the AMP pipeline (pick frames to carve for the reference
turn+cruise motion dataset). Reads only gt.txt (fast, no segmentation/carve)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)



def track(gt_path, fish_id):
    g = np.loadtxt(gt_path, delimiter=",")
    g = g[g[:, 1] == fish_id]
    g = g[np.argsort(g[:, 0])]
    return g[:, 0].astype(int), g[:, 2:5].astype(np.float64) * 0.01  # frames, pos(m)


def smooth(pos, win):
    """Centered moving-average smooth of the (noisy) head track before differentiating."""
    if win <= 1:
        return pos
    k = np.ones(win) / win
    pad = win // 2
    out = np.empty_like(pos)
    for c in range(pos.shape[1]):
        p = np.pad(pos[:, c], pad, mode="edge")
        out[:, c] = np.convolve(p, k, mode="same")[pad:-pad] if pad else np.convolve(p, k, mode="same")
    return out


def kinematics(frames, pos, fps, smooth_win=9):
    dt = 1.0 / fps
    pos = smooth(pos, smooth_win)                             # de-noise head track first
    vel = np.gradient(pos, axis=0) / dt                       # (T,3) m/s
    speed = np.linalg.norm(vel, axis=1)
    # horizontal heading + signed yaw-rate (dominant turning plane)
    heading = np.arctan2(vel[:, 1], vel[:, 0])                # rad
    dheading = np.diff(heading)
    dheading = (dheading + np.pi) % (2 * np.pi) - np.pi       # wrap to [-pi,pi]
    yaw_rate = np.concatenate([[0.0], dheading]) / dt         # rad/s (signed)
    # full 3D turn rate magnitude (any plane)
    u = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9)
    cosang = np.clip(np.sum(u[1:] * u[:-1], axis=1), -1, 1)
    turn3d = np.concatenate([[0.0], np.arccos(cosang)]) / dt  # rad/s (unsigned)
    return speed, yaw_rate, turn3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=_P("${ZEF_ROOT}/data/ZebraFish-01/gt/gt.txt"))
    ap.add_argument("--fish-id", type=int, default=1)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--turn-thresh", type=float, default=1.0, help="|yaw_rate| rad/s above = turning")
    ap.add_argument("--min-speed", type=float, default=0.03, help="ignore near-stationary frames (heading unreliable)")
    ap.add_argument("--smooth-win", type=int, default=9, help="head-track smoothing window (frames)")
    ap.add_argument("--window", type=int, default=120, help="window (frames) to rank turn-richness")
    args = ap.parse_args()

    frames, pos = track(Path(args.gt), args.fish_id)
    speed, yaw, turn3d = kinematics(frames, pos, args.fps, args.smooth_win)
    moving = speed > args.min_speed
    is_turn = (np.abs(yaw) > args.turn_thresh) & moving

    print(f"seq {Path(args.gt).parts[-3]} id {args.fish_id}: {len(frames)} frames "
          f"[{frames.min()}..{frames.max()}], fps {args.fps}")
    print(f"  speed  m/s : mean {speed[moving].mean():.3f}  p50 {np.percentile(speed[moving],50):.3f}  "
          f"p95 {np.percentile(speed[moving],95):.3f}")
    print(f"  |yaw| rad/s: mean {np.abs(yaw[moving]).mean():.2f}  p50 {np.percentile(np.abs(yaw[moving]),50):.2f}  "
          f"p95 {np.percentile(np.abs(yaw[moving]),95):.2f}")
    print(f"  turn frames (|yaw|>{args.turn_thresh} & moving): {is_turn.sum()} "
          f"({100*is_turn.mean():.1f}%)   straight moving: {(moving & ~is_turn).sum()}")

    # rank windows by turn density
    w = args.window
    dens = np.array([is_turn[i:i + w].mean() for i in range(0, len(frames) - w, w)])
    order = np.argsort(dens)[::-1][:8]
    print(f"  top turn-rich {w}-frame windows (start_frame : turn_fraction):")
    for i in sorted(order):
        s = frames[i * w]
        print(f"    {s:6d}-{s + w:6d} : {dens[i]:.2f}")


if __name__ == "__main__":
    main()
