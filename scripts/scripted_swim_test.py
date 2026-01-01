"""THRUST CEILING TEST: drive the fish with a clean scripted traveling wave (open loop, no policy)
and measure the forward speed it reaches, plus the real tail-tip amplitude.

Separates "the POLICY is bad" from "the FISH physically cannot swim fast". Each of the 3 DOF families
(rotX/rotY/rotZ of the D6 joints) is driven separately, because only one of them bends the body
laterally and we should not assume which.
"""

import argparse
import sys
from pathlib import Path

for _p in (Path(_P("${FISH_ROOT}/source")),):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--freqs", type=float, nargs="+", default=[1.5, 2.5])
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--amp", type=float, default=1.0, help="action amplitude (1.0 = full range)")
parser.add_argument("--only_dofs", type=str, nargs="*", default=[], help="restrict to these DOF families")
parser.add_argument("--phase_signs", type=float, nargs="+", default=[1.0], help="+1 and/or -1: wave travel direction")
parser.add_argument("--gui", action="store_true", help="open the Isaac Sim viewport (camera stays manual)")
parser.add_argument("--follow_cam", action="store_true",
                    help="OPT-IN auto camera that chases the fish. Off by default so the viewport "
                         "camera stays yours to drive -- the script never fights your mouse.")
parser.add_argument("--cam_dist", type=float, default=1.2, help="follow-camera distance (m)")
parser.add_argument("--stiffness", type=float, default=0.0,
                    help="override the D6 drive stiffness (0 = keep the cfg value, 60). The drive is a "
                         "first-order lag with corner = stiffness/damping, so this directly sets how "
                         "much of the commanded angle the joint reaches at swimming frequency.")
parser.add_argument("--damping", type=float, default=0.0, help="override drive damping (0 = keep, 6)")
parser.add_argument("--tail_area_scale", type=float, default=1.0,
                    help="multiply the PANEL AREA of the rearmost --tail_frac of the body. Thrust ~ "
                         "area, so this enlarges the propulsor in the hydro proxy only -- no mesh, "
                         "FEM or skeleton change. Note it also raises that region's forward drag.")
parser.add_argument("--tail_frac", type=float, default=0.15,
                    help="rear fraction of body length treated as the caudal fin")
parser.add_argument("--no_deformable", action="store_true",
                    help="disable the FEM soft body -- it is pinned to the bones but the hydro NEVER "
                         "reads it, so it can only absorb actuator work, never make thrust")
parser.add_argument("--wavelengths", type=float, nargs="+", default=[1.0],
                    help="how many full waves fit along the body (real fish run ~0.7-1.0)")
parser.add_argument("--envelopes", type=str, nargs="+", default=["uniform"],
                    help="amplitude envelope along the body: uniform | taper_tail | taper_head. "
                         "Real fish grow amplitude toward the TAIL and keep the head nearly still.")
parser.add_argument("--joint_limit_deg", type=float, nargs="+", default=[0.0],
                    help="Widen (or narrow) every controlled D6 rotation limit to +/-N degrees and "
                         "raise pos_action_scale to match, so amp=1.0 really commands the full range. "
                         "0 = leave the asset's own limit alone. Pass several to A/B them.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
app = AppLauncher(args).app

import itertools  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import FISH.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


TASK = "Template-Salmon-Swim-AMP-Misty-Panels-Direct-v0"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
cfg.pos_action_scale = 0.26          # match the training run's override
if args.no_deformable:
    cfg.with_deformable = False
env = gym.make(TASK, cfg=cfg).unwrapped
dev = env.device
if args.no_deformable:
    # the AMP style reward reads the FEM mesh; this is an open-loop thrust test so rewards are
    # irrelevant -- stub them out rather than drag the soft body along just to satisfy the reward.
    env._get_rewards = lambda: torch.zeros(env.num_envs, device=env.device)
    print("### FEM soft body DISABLED (AMP reward stubbed)")

if args.tail_area_scale != 1.0:
    _d = np.load(env.cfg.panel_hydro_path)
    _bq, _bp = _d['bone_rest_quat'].astype(np.float64), _d['bone_rest_pos'].astype(np.float64)
    _bi, _r = _d['panel_bone'], _d['panel_r_local'].astype(np.float64)
    _w, _x, _y, _z = (_bq[_bi][:, 0], _bq[_bi][:, 1], _bq[_bi][:, 2], _bq[_bi][:, 3])
    _R = np.stack([1-2*(_y*_y+_z*_z), 2*(_x*_y-_w*_z), 2*(_x*_z+_w*_y),
                   2*(_x*_y+_w*_z), 1-2*(_x*_x+_z*_z), 2*(_y*_z-_w*_x),
                   2*(_x*_z-_w*_y), 2*(_y*_z+_w*_x), 1-2*(_x*_x+_y*_y)], -1).reshape(-1, 3, 3)
    _pw = _bp[_bi] + np.einsum('nij,nj->ni', _R, _r)
    _L = float(getattr(env, '_body_length', 0.496))
    _s = (_pw[:, 0].max() - _pw[:, 0]) / _L                # 0 = nose, 1 = tail tip
    _mask = torch.tensor(_s >= (1.0 - args.tail_frac), device=dev)
    env._panel_a = torch.where(_mask, env._panel_a * args.tail_area_scale, env._panel_a)
    print(f"### tail panels (rear {args.tail_frac:.0%}): {int(_mask.sum())} panels area x{args.tail_area_scale} "
          f"-> total wetted {float(env._panel_a.sum()):.4f} m^2")
names = [env.robot.data.joint_names[i] for i in env._control_joint_ids]
nact = len(names)
print(f"\n### {nact} controlled joints: {names}")
L = float(getattr(env, "_body_length", 0.496))
print(f"### body_length={L:.4f}  pos_action_scale={env.cfg.pos_action_scale}  cd_t={env._panel_cd_t}")

# Joints are named "D6Joint_NN:D" -- NN = position along the chain, D = which of the 3 D6 rotation
# DOFs. Group by D (3 families of 7), and order each family head->tail by NN so the phase lag makes a
# real travelling wave.
def _chain_idx(nm):
    base = nm.split(":")[0]
    return int(base.split("_")[-1]) if "_" in base else 0

groups = {}
for i, n in enumerate(names):
    groups.setdefault(n.split(":")[-1], []).append(i)
for k in groups:
    groups[k] = sorted(groups[k], key=lambda i: _chain_idx(names[i]))
print(f"### DOF families: { {k: [names[i] for i in v] for k, v in groups.items()} }")

bpos = env.robot.data.body_link_pos_w[0].clone()          # bone positions, env 0
order = torch.argsort(bpos[:, 0], descending=True)        # head (+x) first

_base_limit_deg = float(env._soft_joint_limits[..., 1].abs().max()) * 57.29578
_base_action_scale = float(env.cfg.pos_action_scale)
print(f"### asset joint limit = +/-{_base_limit_deg:.1f} deg")


def set_joint_limit(deg):
    """Widen the D6 rotation limits AND pos_action_scale together. BOTH gate the command: the env
    clamps targets to _soft_joint_limits, and scales actions by pos_action_scale -- raising only one
    of them changes nothing."""
    if deg <= 0:
        env.cfg.pos_action_scale = _base_action_scale
        return _base_limit_deg
    rad = float(np.deg2rad(deg))
    lim = torch.zeros(env.num_envs, len(env._control_joint_ids), 2, device=dev)
    lim[..., 0], lim[..., 1] = -rad, rad
    env.robot.write_joint_position_limit_to_sim(lim, joint_ids=env._control_joint_ids)
    env._soft_joint_limits = env.robot.data.soft_joint_pos_limits
    env.cfg.pos_action_scale = rad
    got = float(env._soft_joint_limits[..., 1].abs().max()) * 57.29578
    print(f"### joint limit -> +/-{got:.1f} deg (asked {deg:.1f}), pos_action_scale -> {rad:.4f}")
    return got


if args.stiffness > 0 or args.damping > 0:
    _k = args.stiffness if args.stiffness > 0 else float(env.robot.data.joint_stiffness[0, 0])
    _c = args.damping if args.damping > 0 else float(env.robot.data.joint_damping[0, 0])
    env.robot.write_joint_stiffness_to_sim(_k, joint_ids=env._control_joint_ids)
    env.robot.write_joint_damping_to_sim(_c, joint_ids=env._control_joint_ids)
    print(f"### drive gains -> stiffness {_k:.1f}, damping {_c:.1f} "
          f"(first-order corner k/c = {_k/_c:.2f} rad/s = {_k/_c/6.2832:.2f} Hz)")

DT = env.cfg.decimation * env.cfg.sim.dt
n_steps = int(args.seconds / DT)
print(f"### control dt={DT:.4f}s -> {n_steps} steps per trial\n")
rows = []

GROUPS = [(g, i) for g, i in sorted(groups.items()) if (not args.only_dofs or g in args.only_dofs)]
for gname, idxs in GROUPS:
  for psign in args.phase_signs:
    for f, jl, wl, envn in itertools.product(args.freqs, args.joint_limit_deg,
                                             args.wavelengths, args.envelopes):
            lim_deg = set_joint_limit(jl)
            # NOTE: do NOT call env.reset() between trials -- with the FEM body active that reliably
            # triggers a CUDA illegal-memory-access here. A zero-action pause settles it instead.
            for _ in range(45):
                env.step(torch.zeros(env.num_envs, nact, device=dev))
            p0 = env.robot.data.root_state_w[:, 0:3].clone()
            fs = float(getattr(env.cfg, "body_forward_sign", -1.0))
            fwd_b = torch.tensor([fs, 0.0, 0.0], device=dev).repeat(env.num_envs, 1)
            fwd0 = math_utils.quat_apply(env.robot.data.root_state_w[:, 3:7], fwd_b)
            fwd0 = fwd0 / fwd0.norm(dim=1, keepdim=True).clamp(min=1e-6)
            # phase lag along the body -> a travelling wave from head to tail (1 full wave over the body)
            N = len(idxs)
            ph = torch.tensor([2 * np.pi * wl * (k / max(1, N - 1)) for k in range(N)], device=dev)
            # chain index 0 is the TAIL end (verified: psign=-1 makes the wave run head->tail and
            # swims the fish forward), so "taper_tail" must grow the amplitude toward index 0.
            kk = torch.arange(N, device=dev, dtype=torch.float32) / max(1, N - 1)
            if envn == "taper_tail":
                envv = (1.0 - kk) ** 2
            elif envn == "taper_head":
                envv = kk ** 2
            else:
                envv = torch.ones(N, device=dev)
            envv = envv / envv.max()
            vfwd, tipamp, jrec, vlat = [], [], [], []
            for t in range(n_steps):
                a = torch.zeros(env.num_envs, nact, device=dev)
                wave = args.amp * envv * torch.sin(2 * np.pi * f * (t * DT) - psign * ph)
                a[:, idxs] = wave.unsqueeze(0)
                env.step(a)
                st = env.robot.data.root_state_w
                if args.follow_cam and t % 3 == 0:   # opt-in only; default leaves the camera to you
                    c = st[0, 0:3].detach().cpu().numpy()
                    env.sim.set_camera_view(
                        eye=(float(c[0]), float(c[1]) - args.cam_dist, float(c[2]) + 0.45 * args.cam_dist),
                        target=(float(c[0]), float(c[1]), float(c[2])))
                if t > n_steps // 2:                                  # steady window only
                    # project on the CURRENT heading, not the trial-start heading: a fish that turns
                    # is still swimming, and a fixed reference axis scores it as if it had stopped.
                    fwd_now = math_utils.quat_apply(st[:, 3:7], fwd_b)
                    fwd_now = fwd_now / fwd_now.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    vfwd.append(((st[:, 7:10] * fwd_now).sum(-1)).mean().item())
                    bp = env.robot.data.body_link_pos_w[0]
                    head, tail = bp[order[0]], bp[order[-1]]
                    axis = fwd0[0]
                    d = tail - head
                    tipamp.append(float(torch.linalg.norm(d - (d @ axis) * axis)))
                    # ACHIEVED joint angles of the driven DOFs (env 0) -> tracking vs command
                    jrec.append(env.robot.data.joint_pos[0, env._control_joint_ids][idxs]
                                .detach().cpu().numpy().copy())
                    # tail-tip LATERAL speed = what actually pushes water (thrust ~ v^2)
                    vt = env.robot.data.body_link_lin_vel_w[0, order[-1]]
                    vlat.append(float(torch.linalg.norm(vt - (vt @ axis) * axis)))
            J = np.array(jrec)                       # (T, n_joints) achieved angles
            ach = float(np.mean(np.percentile(J, 95, axis=0) - np.percentile(J, 5, axis=0)) / 2)
            cmd = float(args.amp * env.cfg.pos_action_scale * float(envv.mean()))
            vlat_rms = float(np.sqrt(np.mean(np.array(vlat) ** 2)))
            print(f"      joint amp: commanded {cmd:.4f} rad ({np.rad2deg(cmd):.1f} deg) -> "
                  f"ACHIEVED {ach:.4f} rad ({np.rad2deg(ach):.1f} deg) = {100*ach/max(cmd,1e-9):.0f}% tracking"
                  f" | tail-tip lateral speed rms = {vlat_rms:.3f} m/s")
            v = float(np.mean(vfwd))
            amp = float(np.percentile(tipamp, 95) - np.percentile(tipamp, 5))
            rows.append((f"lim{lim_deg:.0f} wl{wl:g} {envn}", f, v, v / L, amp, amp / L))
            print(f"  lim{lim_deg:.0f} wl{wl:g} {envn:10s} @ {f:.1f}Hz : v_fwd = {v:+.4f} m/s = {v/L:+.4f} BL/s | "
                  f"tail sway = {amp*100:.1f} cm = {amp/L:.3f} BL")

print("\n=== SUMMARY (real fish: ~1-4 BL/s, tail sway ~0.1 BL) ===")
for g, f, v, vb, a, ab in sorted(rows, key=lambda r: -abs(r[3])):
    print(f"  {g} {f:.1f}Hz  {vb:+.4f} BL/s   tail {ab:.3f} BL")
print("THRUST_TEST_DONE")
