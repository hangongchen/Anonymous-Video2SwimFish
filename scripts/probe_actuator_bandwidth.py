#!/usr/bin/env python
"""Open-loop actuator BANDWIDTH probe: can the FEM fish physically undulate at ~3 Hz?

The AMP reference tail beat is ~2-3 Hz. If the PD joint drives cannot swing that fast against FEM
inertia + water drag, the discriminator can never be satisfied and no reward tuning helps -- so this
is a hard prerequisite gate BEFORE training (per the task spec).

Drives every joint with a body-traveling-wave POSITION target a*sin(2*pi*f*t + phase[bone]) at
f = 0.5/1/2/3/4 Hz, and per frequency measures:
  - TRACKING ratio = achieved joint oscillation amplitude / commanded amplitude (drops toward 0 if the
    actuator cannot keep up -> the bandwidth limit).
  - TAIL amplitude in the world-horizontal swim plane (does it actually produce undulation).
At the 3 Hz drive it ALSO logs the reward penalties (energy/jvel/action-rate/depth) to confirm they do
NOT suppress a vigorous gait (they must stay << r_style ~ 0.75).

  FISH_TEST_DT_HZ=120 HYDRO_TORQUE_CLIP=0.05 <isaac python> scripts/probe_actuator_bandwidth.py
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--amp", type=float, default=0.7, help="commanded action amplitude in [-1,1]")
ap.add_argument("--secs", type=float, default=4.0, help="seconds per frequency")
ap.add_argument("--freqs", type=str, default="0.5,1,2,3,4")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

sys.path.insert(0, "source/FISH")
import FISH.tasks.direct.fish  # noqa: E402,F401
from FISH.tasks.direct.fish.salmon_amp_autoskel_cfg import SalmonAMPAutoskelCfg  # noqa: E402

cfg = SalmonAMPAutoskelCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.debug_print = False
env = gym.make("Template-Salmon-AMP-Autoskel-Direct-v0", cfg=cfg)
base = env.unwrapped
dt_ctrl = float(cfg.sim.dt) * float(cfg.decimation)
nb = base._num_actions // 3                                   # bones (3 DOF each)
pos_scale = float(cfg.pos_action_scale)
print(f"[probe] control dt={dt_ctrl*1000:.2f} ms ({1/dt_ctrl:.0f} Hz), {base._num_actions} DOF / {nb} bones, "
      f"pos_action_scale={pos_scale} rad, commanded amp={args.amp} -> {args.amp*pos_scale:.3f} rad", flush=True)

freqs = [float(x) for x in args.freqs.split(",")]
# body-traveling-wave phase per BONE (one full wave down the body), broadcast to its 3 DOF
bone_phase = torch.linspace(0, 2 * math.pi, nb, device=base.device)
phase = bone_phase.repeat_interleave(3)[: base._num_actions]


def hstack_tail_amp(pc_hist, root_hist):
    """world-horizontal, heading-relative tail-station lateral amplitude (BL) over a window."""
    lat_tail = []
    for pc, root in zip(pc_hist, root_hist):
        rel = pc - root[0:3]
        q = root[3:7]
        w, x, y, z = q
        Rt = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        fwd = Rt @ np.array([-1.0, 0.0, 0.0]); fwd[2] = 0
        fwd = fwd / (np.linalg.norm(fwd[:2]) + 1e-9)
        perp = np.array([-fwd[1], fwd[0], 0.0])
        along = rel @ fwd
        tail = along > np.percentile(along, 80)
        lat_tail.append((rel[tail] @ perp).mean())
    return float(np.std(lat_tail)) / max(float(cfg.body_length), 1e-6)


base.reset()
results = []
pen_report = None
for f in freqs:
    nsteps = int(args.secs / dt_ctrl)
    jp_hist, pc_hist, root_hist = [], [], []
    logs = {k: [] for k in ("energy", "jvel_pen", "r_style", "pen_z", "v_forward_bl")}
    t0 = 0.0
    for t in range(nsteps):
        tt = t * dt_ctrl
        act = args.amp * torch.sin(2 * math.pi * f * tt + phase).to(base.device)
        act = act.unsqueeze(0).expand(base.num_envs, -1).contiguous()
        base.step(act)
        if t > nsteps // 3:                                   # steady window (skip transient)
            jp_hist.append(base.robot.data.joint_pos[0].detach().cpu().numpy().copy())
            _np = base._soft_view.get_simulation_mesh_nodal_positions() if hasattr(base, "_soft_view") else None
            pc_hist.append(np.asarray(_np[0].detach().cpu().numpy() if hasattr(_np, "detach") else _np[0])
                           if _np is not None else None)
            root_hist.append(base.robot.data.root_state_w[0].detach().cpu().numpy().copy())
            log = base.extras.get("log", {})
            for k in logs:
                if k in log:
                    logs[k].append(float(log[k]))
    jp = np.array(jp_hist)                                    # (T, nDOF)
    # achieved per-DOF oscillation amplitude (robust: sqrt(2)*std) vs commanded
    achieved = np.sqrt(2) * jp.std(0)
    commanded = args.amp * pos_scale
    track = np.median(achieved) / max(commanded, 1e-9)
    tail_amp = hstack_tail_amp(pc_hist, root_hist) if pc_hist and pc_hist[0] is not None else float("nan")
    results.append((f, track, tail_amp, float(np.median(achieved))))
    print(f"[probe] f={f:.1f} Hz: tracking={track:.2f} (achieved {np.median(achieved):.3f} / "
          f"commanded {commanded:.3f} rad), tail_amp={tail_amp:.3f} BL", flush=True)
    if abs(f - 3.0) < 1e-6:
        pen_report = {k: (np.mean(v) if v else float("nan")) for k, v in logs.items()}

print("\n=== ACTUATOR BANDWIDTH SUMMARY ===")
for f, track, tail_amp, ach in results:
    bar = "#" * int(track * 40)
    print(f"  {f:>4.1f} Hz | tracking {track:5.2f} | tail {tail_amp:5.3f} BL | {bar}")
r3 = next((r for r in results if abs(r[0] - 3.0) < 1e-6), None)
reach3 = r3 is not None and r3[1] > 0.4 and r3[2] > 0.01
print(f"\n  ~3 Hz reachable: {'YES' if reach3 else 'NO'}  "
      f"(tracking {r3[1]:.2f} > 0.4 and tail {r3[2]:.3f} > 0.01 BL)" if r3 else "  no 3 Hz sample")
if pen_report:
    print("\n=== reward penalties DURING the 3 Hz gait (must be << r_style ~0.75) ===")
    for k, v in pen_report.items():
        print(f"  {k:14s} {v:.4f}")
    tot = (float(cfg.w_energy) * pen_report.get("energy", 0)
           + float(cfg.w_jvel) * pen_report.get("jvel_pen", 0)
           + float(cfg.w_zband) * pen_report.get("pen_z", 0))
    print(f"  weighted penalty sum (energy+jvel+z) = {tot:.4f}  "
          f"({'OK: << r_style' if tot < 0.2 else 'WARNING: may suppress gait'})")
print(f"\nPROBE_VERDICT: {'PASS_3HZ_REACHABLE' if reach3 else 'FAIL_3HZ_UNREACHABLE'}", flush=True)
os._exit(0)
