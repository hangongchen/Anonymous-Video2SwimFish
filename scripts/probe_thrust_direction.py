#!/usr/bin/env python
"""Definitive physics test: can ANY clean undulatory gait make the fish swim FORWARD (head-first, -X)?

Six AMP training runs all settled into a tail-first backward drift regardless of the reward. Before
blaming the policy, test the PHYSICS directly: drive a clean LATERAL (yaw-plane) traveling wave in
both directions and measure which way the fish actually translates.

  1. Identify the yaw DOF: hold each of the 3 DOF per bone as a static bend, measure which produces
     world-HORIZONTAL tail deflection (that DOF is the lateral/undulation one).
  2. Drive a head->tail and a tail->head lateral traveling wave on that DOF, measure v_forward
     (= -root_lin_vel_b[:,0]; head is at -X).
     v_forward > 0  -> the fish swims FORWARD for that wave direction => forward swimming IS possible
                       (the training failure is a learning/exploration problem).
     v_forward <= 0 for BOTH -> the hydro cannot produce forward undulatory thrust (a physics problem).

  FISH_OPEN_WATER=1 FISH_TEST_DT_HZ=120 HYDRO_TORQUE_CLIP=0.05 <isaac py> scripts/probe_thrust_direction.py
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=3)
ap.add_argument("--amp", type=float, default=0.9)
ap.add_argument("--freq", type=float, default=2.0)
ap.add_argument("--secs", type=float, default=5.0)
ap.add_argument("--ck", type=float, default=None, help="override Kutta lift coefficient")
ap.add_argument("--force-clip", type=float, default=None, help="override hydro force clip (N)")
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
if args.ck is not None:
    cfg.ck = args.ck
    print(f"[probe] ck (Kutta lift) OVERRIDE -> {cfg.ck}", flush=True)
if args.force_clip is not None:
    cfg.hydro_force_clip = args.force_clip
    print(f"[probe] hydro_force_clip OVERRIDE -> {cfg.hydro_force_clip}", flush=True)
env = gym.make("Template-Salmon-AMP-Autoskel-Direct-v0", cfg=cfg)
base = env.unwrapped
dt = float(cfg.sim.dt) * float(cfg.decimation)
nb = base._num_actions // 3
pos_scale = float(cfg.pos_action_scale)


def drive(nsteps, act_fn, settle=0.4):
    """Drive for nsteps WITHOUT resetting; return (mean, median) v_forward over the steady tail."""
    vs = []
    for t in range(nsteps):
        tt = t * dt
        act = act_fn(tt).unsqueeze(0).expand(base.num_envs, -1).contiguous()
        base.step(act)
        if t > settle * nsteps:
            vf = -base.robot.data.root_lin_vel_b[:, 0]        # body -X = head-first forward
            vs.append(float(vf.mean()))
    return float(np.mean(vs)), float(np.median(vs))


# ---- 1. identify the yaw (lateral) DOF: static bend on each of the 3 DOF, measure horizontal tail defl
print("[probe] identifying the lateral/yaw DOF (of 3 per bone)...", flush=True)
dof_lateral = {}
for dof in range(3):
    base.reset()
    a = torch.zeros(base._num_actions, device=base.device)
    idx = torch.arange(base._num_actions, device=base.device)
    a[idx % 3 == dof] = 0.8                                   # static bend on this DOF across all bones
    for _ in range(30):
        base.step(a.unsqueeze(0).expand(base.num_envs, -1).contiguous())
    pc = base._soft_view.get_simulation_mesh_nodal_positions()[0].detach().cpu().numpy()
    root = base.robot.data.root_state_w[0].detach().cpu().numpy()
    rel = pc - root[0:3]
    w, x, y, z = root[3:7]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    fwd = R @ np.array([-1., 0, 0]); fwd[2] = 0; fwd /= np.linalg.norm(fwd[:2]) + 1e-9
    perp = np.array([-fwd[1], fwd[0], 0.])
    along = rel @ fwd
    tail = along > np.percentile(along, 80)
    horiz_defl = abs((rel[tail] @ perp).mean())
    vert_defl = abs((rel[tail][:, 2]).mean())
    dof_lateral[dof] = horiz_defl
    print(f"  DOF {dof}: horizontal tail defl {horiz_defl:.4f}  vertical {vert_defl:.4f}", flush=True)
yaw_dof = max(dof_lateral, key=dof_lateral.get)
print(f"  -> lateral/yaw DOF = {yaw_dof} (max horizontal deflection)", flush=True)

# ---- 2. traveling waves on the yaw DOF, both directions ----
nsteps = int(args.secs / dt)
bone_phase = torch.linspace(0, 2 * math.pi, nb, device=base.device)      # 0 at head bone -> 2pi at tail
phase = torch.zeros(base._num_actions, device=base.device)
phase[torch.arange(base._num_actions, device=base.device) % 3 == yaw_dof] = bone_phase.repeat_interleave(3)[
    torch.arange(base._num_actions, device=base.device) % 3 == yaw_dof]
mask = (torch.arange(base._num_actions, device=base.device) % 3 == yaw_dof).float()


def wave(sign):
    # sign=+1: sin(wt + phase) -> crest moves toward LOWER phase = toward HEAD = tail->head wave
    # sign=-1: sin(wt - phase) -> crest moves toward HIGHER phase = toward TAIL = head->tail wave
    def f(tt):
        return args.amp * mask * torch.sin(2 * math.pi * args.freq * tt + sign * phase)
    return f


def envelope():
    """tail-weighted amplitude per bone (real fish: amplitude grows toward the tail)."""
    s = torch.linspace(0, 1, nb, device=base.device)
    env = 0.15 + 0.85 * s ** 2
    return env.repeat_interleave(3)[torch.arange(base._num_actions, device=base.device)]


BL = float(cfg.body_length)
env_amp = envelope()


def wave_env(sign, use_env):
    amp_vec = args.amp * (env_amp if use_env else torch.ones_like(env_amp))

    def f(tt):
        return amp_vec * mask * torch.sin(2 * math.pi * args.freq * tt + sign * phase)
    return f


def net_forward_displacement(nsteps, act_fn, settle=0.3):
    """NET COM displacement along body-forward (-X world), averaged over envs, in metres.
    Uses displacement (robust) not instantaneous velocity."""
    base.reset()
    p0 = None
    t_start = int(settle * nsteps)
    for t in range(nsteps):
        act = act_fn(t * dt).unsqueeze(0).expand(base.num_envs, -1).contiguous()
        base.step(act)
        if t == t_start:
            p0 = base.robot.data.root_state_w[:, 0:3].clone()
            q0 = base.robot.data.root_state_w[:, 3:7].clone()
    p1 = base.robot.data.root_state_w[:, 0:3]
    disp = p1 - p0
    # forward = body -X in world (use the start heading)
    fwd = math_utils.quat_apply(q0, torch.tensor([-1.0, 0.0, 0.0], device=base.device).expand(base.num_envs, 3))
    fwd_h = fwd.clone(); fwd_h[:, 2] = 0
    fwd_h = fwd_h / torch.linalg.norm(fwd_h[:, :2], dim=-1, keepdim=True).clamp(min=1e-6)
    fwd_disp = (disp * fwd_h).sum(-1)
    total = torch.linalg.norm(disp[:, :2], dim=-1)
    return float(fwd_disp.mean()), float(total.mean())


import isaaclab.utils.math as math_utils  # noqa: E402
FREQS = [0.5, 1.0, 1.5, 2.0, 3.0]
secs = 6.0
ns = int(secs / dt)
print(f"\n[probe] frequency sweep, NET forward displacement over {secs}s, amp {args.amp}", flush=True)
print(f"  {'freq':>5} | {'H->T uniform':>13} {'H->T tailwt':>12} {'T->H uniform':>13}  (forward metres, +=head-first)")
results = []
for f in FREQS:
    args.freq = f
    fd_htu, _ = net_forward_displacement(ns, wave_env(-1, False))
    fd_htw, _ = net_forward_displacement(ns, wave_env(-1, True))
    fd_thu, _ = net_forward_displacement(ns, wave_env(+1, False))
    results.append((f, fd_htu, fd_htw, fd_thu))
    print(f"  {f:>5.1f} | {fd_htu:>+13.4f} {fd_htw:>+12.4f} {fd_thu:>+13.4f}", flush=True)
best = max(max(r[1], r[2], r[3]) for r in results) / BL / secs   # best forward speed (BL/s)
print(f"\n  best NET FORWARD speed by ANY clean undulatory wave: {best:+.3f} BL/s over {secs}s")
verdict = ("FORWARD_POSSIBLE (undulation gives head-first thrust -> the training/exploration was the problem)"
           if best > 0.03 else
           "FORWARD_IMPOSSIBLE (no undulatory wave gives forward thrust -> the per-slice HYDRO cannot "
           "propel the fish by undulation; only crabbing/paddling moves it, as the old policy did)")
print(f"PROBE_VERDICT: {verdict}", flush=True)
os._exit(0)
