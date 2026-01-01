#!/usr/bin/env python
"""Step 4 of the AMP pipeline: EVALUATE a trained reactive-swimming policy.

Offline analyzer of a recorded rollout (from `scripts/rl_games/play.py --record_demo <npz>` on the
AMP task -- arrays: pc (T,N,3) world FEM cloud, root (T,13) world pose). Computes the eval metrics
from AMP_TANK_SWIM_PIPELINE.md: cruising, gait/turn realism vs the real ZeF reference, runs-forever,
reactivity to the tank walls, and efficiency. No Isaac needed.

  python scripts/eval_amp_swim.py --demo <rollout.npz> --reference demo_out/zef_amp_ref/amp_reference.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_amp_features import spine_bend_profile

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


# quaternion (w,x,y,z) inverse-rotate a set of vectors (numpy)
def quat_apply_inverse(q, v):
    w, x, y, z = q
    # rotation matrix of q, then transpose (inverse for unit quat)
    R = np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])
    return v @ R  # v (N,3) @ R  == R^T applied to each row (inverse rotate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, help="recorded AMP rollout npz (pc, root)")
    ap.add_argument("--reference", default=_P("${FISH_ROOT}/demo_out/zef_amp_ref/amp_reference.npz"))
    ap.add_argument("--fps", type=float, default=30.0, help="control rate (Hz)")
    ap.add_argument("--body-len", type=float, default=0.5, help="fish body length (m)")
    ap.add_argument("--tank-half", type=float, default=1.0, help="tank half-size (m)")
    ap.add_argument("--wall-margin", type=float, default=0.15)
    ap.add_argument("--profile-len", type=int, default=20)
    args = ap.parse_args()

    d = np.load(args.demo)
    pc = np.asarray(d["pc"], np.float64)          # (T,N,3) world
    root = np.asarray(d["root"], np.float64)      # (T,13) world  [pos3, quat4, linvel3, angvel3]
    T = len(pc); dt = 1.0 / args.fps; K = args.profile_len
    pos = root[:, 0:3]

    # ---- cruising ----
    vel = np.gradient(pos, axis=0) / dt
    speed = np.linalg.norm(vel[:, 0:2], axis=1)                 # horizontal m/s
    cruise = speed[T // 5:].mean()                             # steady-state (drop first 20%)

    # ---- body-frame bend profile per frame (same extractor as the reference) ----
    bend = np.zeros((T, K))
    for t in range(T):
        body = quat_apply_inverse(root[t, 3:7], pc[t] - pos[t])
        bend[t] = spine_bend_profile(body, K)
    tail = bend[:, -1]                                          # tail-region lateral bend

    # ---- tail-beat frequency (FFT of tail bend, steady-state) ----
    seg = tail[T // 5:] - tail[T // 5:].mean()
    if len(seg) > 8:
        freqs = np.fft.rfftfreq(len(seg), dt)
        amp = np.abs(np.fft.rfft(seg))
        amp[0] = 0
        f_beat = float(freqs[np.argmax(amp)])
    else:
        f_beat = float("nan")
    tail_amp = float(np.percentile(np.abs(seg), 95)) if len(seg) else float("nan")
    strouhal = f_beat * tail_amp * args.body_len / max(cruise, 1e-6)  # tail_amp is body-len-normalized

    # ---- gait realism: bend-transition distribution distance to the real reference ----
    ref = np.load(args.reference)
    rfeat = ref["feat"][:, :K]; rseg = ref["segment_id"]
    rpair = rseg[1:] == rseg[:-1]
    ref_tr = np.concatenate([rfeat[:-1][rpair], rfeat[1:][rpair]], axis=1)   # (P,2K)
    pol_tr = np.concatenate([bend[:-1], bend[1:]], axis=1)                   # (T-1,2K)
    mu, sd = ref_tr.mean(0), ref_tr.std(0) + 1e-4
    ref_n = (ref_tr - mu) / sd; pol_n = (pol_tr - mu) / sd
    # mean nearest-neighbour distance policy->reference (lower = more real)
    sub = pol_n[:: max(1, len(pol_n) // 500)]
    nn = np.array([np.min(np.linalg.norm(ref_n - p, axis=1)) for p in sub])
    gait_realism_nn = float(nn.mean())

    # ---- reactivity to walls (root pos is world; env0 origin ~ 0 so ~ env-local, tank-centred) ----
    clear = args.tank_half - np.abs(pos[:, 0:2])               # clearance to each axis wall
    min_clear = clear.min(axis=1)
    wall_contact_frac = float((min_clear < args.wall_margin).mean())
    coverage = float((np.ptp(pos[:, 0]) * np.ptp(pos[:, 1])) / (2 * args.tank_half) ** 2)

    # ---- runs-forever / stability ----
    finite = np.isfinite(pos).all(1)
    survived = int(finite.sum())
    speed_nondecay = float(speed[-T // 5:].mean() / max(speed[:T // 5].mean(), 1e-6))

    print(f"=== AMP swim eval ({T} steps, {T*dt:.0f}s) ===")
    print(f"cruising      : {cruise:.3f} m/s  ({cruise/args.body_len:.2f} body-len/s)")
    print(f"tail-beat     : {f_beat:.2f} Hz   tail bend amp {tail_amp:.3f} BL   Strouhal {strouhal:.2f} "
          f"(real fish ~0.2-0.4)")
    print(f"gait realism  : mean NN dist to real bend-transitions {gait_realism_nn:.2f} (lower=more real)")
    print(f"reactivity    : wall-contact frac {wall_contact_frac:.2f}  tank coverage {coverage:.2f}")
    print(f"runs-forever  : survived {survived}/{T} finite steps  speed non-decay {speed_nondecay:.2f} "
          f"(>~0.8 good)")


if __name__ == "__main__":
    main()
