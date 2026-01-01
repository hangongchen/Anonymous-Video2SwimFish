#!/usr/bin/env python
"""Analyze a DETERMINISTIC play.py --record_demo rollout npz for the freeze test.

Reports the decisive numbers the task specifies: deterministic joint_vel, joint amplitude, root path
over the window. Frozen policy (the failure) shows joint_vel median < 0.05 rad/s and ~0 BL travel.

  python scripts/analyze_det_rollout.py <rollout.npz> [--bl 0.5]
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("npz")
ap.add_argument("--bl", type=float, default=0.5)
args = ap.parse_args()

d = np.load(args.npz)
jv = d["joint_vel"]; jp = d["joint_pos"]; root = d["root"]; dt = float(d["step_dt"])
Tfull = len(jv); BL = args.bl
# STEADY window: skip the first 1 s (post-reset transient) and the last 2 s (the hydro-lift LAUNCH
# artifact inflates path/speed -- see hydro-lift-launch-artifact memory). Quote cruise over this window.
lo = min(int(1.0 / dt), Tfull // 4)
hi = max(lo + 1, Tfull - int(2.0 / dt))
jv = jv[lo:hi]; jp = jp[lo:hi]; root = root[lo:hi]
T = len(jv)
pos = root[:, 0:3]
print(f"[window] steady {lo}-{hi} of {Tfull} steps ({T*dt:.1f}s; skipped 1s transient + 2s end-launch)")
# the live lateral DOF is every 3rd (%3==2); report the moving joints, not the 18 locked ones
live = np.arange(jp.shape[1]) % 3 == 2
pp = jp.max(0) - jp.min(0)
disp = np.linalg.norm(pos[-1] - pos[0])
pathlen = np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()
horiz_path = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum()

print(f"=== DETERMINISTIC rollout: {T} steps ({T*dt:.1f}s @ {1/dt:.0f}Hz) ===")
print(f"  joint_vel |.|      median={np.median(np.abs(jv)):.4f}  live-DOF median="
      f"{np.median(np.abs(jv[:, live])):.4f}  max={np.abs(jv).max():.3f} rad/s")
print(f"  joint_pos p2p      all-median={np.median(pp):.4f}  LIVE-DOF median={np.median(pp[live]):.4f}  "
      f"tail(DOF26)={pp[-1]:.4f} rad")
print(f"  root NET disp      {disp:.4f} m ({disp/BL:.3f} BL)   horizontal path={horiz_path:.4f} m "
      f"({horiz_path/BL:.3f} BL)")
print(f"  root mean speed    {pathlen/(T*dt):.4f} m/s ({pathlen/(T*dt)/BL:.3f} BL/s)")
live_p2p = np.median(pp[live])
undulating = (np.median(np.abs(jv[:, live])) > 0.15) and (live_p2p > 0.03)
translating = horiz_path / BL > 0.5
print(f"  VERDICT: {'UNDULATES' if undulating else 'FROZEN'} "
      f"(live jvel {np.median(np.abs(jv[:, live])):.3f}, live p2p {live_p2p:.3f}); "
      f"{'TRANSLATES' if translating else 'in place'} ({horiz_path/BL:.2f} BL horiz)")
