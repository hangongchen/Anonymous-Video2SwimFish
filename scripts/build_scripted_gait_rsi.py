#!/usr/bin/env python
"""Build a FULL-STATE swimming reference for Reference State Initialization (RSI) of the AMP fish.

The deterministic AMP policy FREEZES because every episode resets to the straight, still default
pose: coherent undulatory swimming is a narrow, coordinated region of joint-space, and random
exploration from frozen yields ~0 net thrust while still paying jvel/action penalties -> the
gradient out of "frozen" points downhill first (an EXPLORATION BARRIER, confirmed: ep_10
deterministic joint_vel median 0.009 rad/s, 0.028 BL over 20 s).

RSI is the textbook cure (DeepMimic / AMP): reset the fish to a RANDOM PHASE of a real swimming
motion so the policy only has to CONTINUE a gait, not invent one. But RSI needs full-state frames
(root, joint pos+vel, FEM nodal pos+vel) and we have no swimmer yet to record from -- chicken/egg.

This script breaks that by driving a SCRIPTED open-loop traveling wave (the same head->tail
tail-weighted gait `probe_thrust_direction.py` proved gives forward thrust), letting the FEM
soft body settle into it, then recording the FEM-CONSISTENT full state each control step. RSI to
these frames is blow-up-safe (the nodes are exactly where they settle for that joint pose) and drops
the policy into a MOVING, coordinated gait. The policy still must LEARN to sustain it (the seeded
joint velocity decays under PD-to-policy-target), so this is standard RSI, NOT scripting the answer.

Output npz matches play.py --record_demo:  pc (T,N,3) env0-local world nodal pos, root (T,13),
joint_pos (T,J), joint_vel (T,J), nodal_vel (T,N,3), step_dt.  Load it via cfg.rsi_reference_path.

  FISH_OPEN_WATER=1 FISH_TEST_DT_HZ=120 HYDRO_TORQUE_CLIP=0.05 HYDRO_FORCE_CLIP=5.0 \
    <env_isaaclab>/bin/python scripts/build_scripted_gait_rsi.py --freq 2.0 --amp 0.9 --record_s 3.0
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--amp", type=float, default=0.9, help="peak action amplitude of the scripted wave")
ap.add_argument("--freq", type=float, default=1.5, help="tail-beat frequency (Hz); 1.5 -> tracking 0.73")
ap.add_argument("--settle_s", type=float, default=2.0, help="drive this long before recording (FEM settles)")
ap.add_argument("--record_s", type=float, default=3.0, help="record this many seconds of steady gait")
ap.add_argument("--out", type=str, default="demo_out/scripted_gait_rsi.npz")
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
cfg.with_deformable = True
cfg.reset_deformable = True
env = gym.make("Template-Salmon-AMP-Autoskel-Direct-v0", cfg=cfg)
base = env.unwrapped
dt = float(cfg.sim.dt) * float(cfg.decimation)
nb = base._num_actions // 3
dev = base.device
print(f"[rsi] env up: num_actions={base._num_actions} bones={nb} dt={dt:.5f}s ({1/dt:.0f}Hz ctrl)", flush=True)

if getattr(base, "_soft_view", None) is None:
    raise RuntimeError("RSI recorder needs the deformable soft view (with_deformable=True).")


# ---- traveling wave driving ALL 3 DOF per bone (the actuator-bandwidth probe proved this MOVES the
# joints -- tracking 0.63 @2Hz, 0.73 @1.5Hz -- and produces horizontal tail deflection; masking to a
# single "yaw" DOF drove a nearly-DEAD DOF (joint p2p 0.001 rad) because the static-bend DOF-ID is
# too noisy to isolate the live lateral axis). Per-BONE traveling-wave phase, broadcast to its 3 DOF.
idx = torch.arange(base._num_actions, device=dev)
bone_phase = torch.linspace(0, 2 * math.pi, nb, device=dev)                 # 0 head -> 2pi tail
phase = bone_phase.repeat_interleave(3)                                     # (num_actions,)
env_amp = torch.linspace(0.35, 1.0, nb, device=dev).repeat_interleave(3)     # amplitude grows toward tail


def wave(tt):
    # crest travels head->tail; drives all 3 DOF -> a diagonal bend with a real horizontal (swim-plane)
    # component. RSI only needs a MOVING, undulating seed state; the policy + AMP reward refine the plane.
    amp_vec = args.amp * env_amp
    return amp_vec * torch.sin(2 * math.pi * args.freq * tt - phase)


# ---- 3. settle the FEM into the gait, then record full state each control step ----
base.reset()
n_settle = int(args.settle_s / dt)
n_rec = int(args.record_s / dt)
print(f"[rsi] settling {n_settle} steps ({args.settle_s}s) then recording {n_rec} steps ({args.record_s}s)...",
      flush=True)
for t in range(n_settle):
    act = wave(t * dt).unsqueeze(0).expand(base.num_envs, -1).contiguous()
    base.step(act)

env0_origin = base.scene.env_origins[0].detach().cpu().numpy()               # (3,)
pc_rec, root_rec, jp_rec, jv_rec, nv_rec = [], [], [], [], []
vf_samples = []
for t in range(n_rec):
    tt = (n_settle + t) * dt
    act = wave(tt).unsqueeze(0).expand(base.num_envs, -1).contiguous()
    base.step(act)
    pos = base._soft_view.get_simulation_mesh_nodal_positions()[0].detach().cpu().numpy().copy()
    vel = base._soft_view.get_simulation_mesh_nodal_velocities()[0].detach().cpu().numpy().copy()
    root = base.robot.data.root_state_w[0].detach().cpu().numpy().copy()      # (13,)
    jp = base.robot.data.joint_pos[0].detach().cpu().numpy().copy()
    jv = base.robot.data.joint_vel[0].detach().cpu().numpy().copy()
    # store env0-LOCAL (world - env0 origin), exactly like play.py --record_demo / the IL RSI loader
    root[0:3] -= env0_origin
    pos -= env0_origin[None, :]
    pc_rec.append(pos); root_rec.append(root); jp_rec.append(jp); jv_rec.append(jv); nv_rec.append(vel)
    vf_samples.append(float(-base.robot.data.root_lin_vel_b[0, 0]))           # body -X = forward

pc_rec = np.stack(pc_rec); root_rec = np.stack(root_rec)
jp_rec = np.stack(jp_rec); jv_rec = np.stack(jv_rec); nv_rec = np.stack(nv_rec)
vf = np.array(vf_samples)
BL = float(cfg.body_length)
outp = os.path.abspath(args.out)
os.makedirs(os.path.dirname(outp), exist_ok=True)
np.savez_compressed(outp, pc=pc_rec, root=root_rec, joint_pos=jp_rec, joint_vel=jv_rec,
                    nodal_vel=nv_rec, step_dt=float(dt), freq=float(args.freq))
print(f"\n[rsi] recorded {n_rec} frames (N={pc_rec.shape[1]} nodes, J={jp_rec.shape[1]} joints) -> {outp}",
      flush=True)
print(f"[rsi] scripted-gait forward speed over the recorded window: "
      f"mean {vf.mean():+.4f} m/s ({vf.mean()/BL:+.3f} BL/s)  median {np.median(vf)/BL:+.3f} BL/s", flush=True)
print(f"[rsi] joint_vel |.| median {np.median(np.abs(jv_rec)):.3f} rad/s  "
      f"joint_pos p2p median {np.median(jp_rec.max(0)-jp_rec.min(0)):.3f} rad  "
      f"(vs FROZEN policy 0.009 / 0.040)", flush=True)
verdict = "GOOD RSI SOURCE" if vf.mean() / BL > 0.05 else "WEAK (gait barely swims -- raise amp/freq)"
print(f"RSI_RECORD: {verdict}", flush=True)
os._exit(0)
