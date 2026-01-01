#!/usr/bin/env python
"""Quick stability probe for the per-slice hydro at coarse dt: boot the AMP autoskel env, drive it
with random actions for N control steps, report peak joint velocity + blow-up count.
Verdict: STABLE if no blow-ups and peak jvel stays sane.

  FISH_TEST_DT_HZ=120 HYDRO_TORQUE_CLIP=0.05 <isaac python> scripts/probe_hydro_stability.py
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import sys  # noqa: E402
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
print(f"[probe] dt=1/{round(1/cfg.sim.dt)} torque_clip={cfg.hydro_torque_clip} "
      f"force_clip={cfg.hydro_force_clip} semi_scale={cfg.hydro_semi_scale}", flush=True)
base.reset()
torch.manual_seed(0)
peak_jvel, blowups = 0.0, 0
for t in range(args.steps):
    act = 2.0 * torch.rand((base.num_envs, base._num_actions), device=base.device) - 1.0
    base.step(act)
    jv = float(base.robot.data.joint_vel.abs().max())
    peak_jvel = max(peak_jvel, jv)
    blowups += int(getattr(base, "_blew_now", 0))
print(f"[probe] steps={args.steps} peak_jvel={peak_jvel:.1f} blowups={blowups}", flush=True)
print(f"PROBE_VERDICT: {'STABLE' if blowups == 0 and peak_jvel < 100 else 'UNSTABLE'}", flush=True)
import os  # noqa: E402
os._exit(0)
