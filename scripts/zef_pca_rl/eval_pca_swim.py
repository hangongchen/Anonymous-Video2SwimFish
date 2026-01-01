"""Deterministic evaluation of the 2-PC PCA-manifold PPO policy + the 4-way baseline comparison.

One batched rollout, one env per condition:
  env 0  FROZEN        zero joint targets (coast; the "is anything better than nothing" floor)
  env 1  SCRIPTED WAVE the hand-designed traveling wave on the same 7 lateral joints
                       (the physical reference; ':1' family reached ~0.53 BL/s in scripted tests)
  env 2  ZEF PLAYBACK  the raw ZeF 2-PC coefficient clip fed through the SAME a->q map
  env 3+ POLICY        the trained PPO policy, deterministic (mu), one env per spawn-noise seed

Everything is recorded per control step and analyzed offline:
  forward/lateral body-frame velocity, net yaw + heading error, tail-beat frequency,
  a1/a2 trajectories + phase portrait, commanded vs measured curvature (same estimator as the
  playback validation) + RMSE, joint tracking, energy, and a top-view video per condition.

Usage:
  <env python> scripts/zef_pca_rl/eval_pca_swim.py --checkpoint logs/rl_games/misty_pca_swim/<run>/nn/<file>.pth
Outputs: demo_out/pca_rl_eval/<run-tag>/  (recording npz, metrics.json, plots/, videos/)
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(_P("${FISH_ROOT}"))
for _p in (str(REPO / "source"), str(REPO / "scripts" / "zef_playback")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--n_policy", type=int, default=5, help="policy envs (different spawn noise)")
parser.add_argument("--wave_freq", type=float, default=2.5)
parser.add_argument("--wave_amp_deg", type=float, default=27.0, help="scripted wave joint amplitude")
parser.add_argument("--out", type=str, default=str(REPO / "demo_out/pca_rl_eval"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import yaml                      # noqa: E402

import FISH.tasks                # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_games.torch_runner import Runner        # noqa: E402

import curvature_utils as cu     # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


TASK = "Template-Salmon-Swim-PCA-Misty-Direct-v0"
E_FROZEN, E_WAVE, E_ZEF = 0, 1, 2
N_ENV = 3 + args.n_policy

# ------------------------------------------------------------------------------- env
cfg = parse_env_cfg(TASK, device=args.device, num_envs=N_ENV)
# FINAL-EVAL convention: start from REST (training uses a 0.05 m/s spawn glide as a
# cold-start bootstrap; in this slippery zero-g water that glide coasts for many seconds
# and would flatter every condition's speed numbers).
cfg.spawn_glide_speed = 0.0
env = gym.make(TASK, cfg=cfg).unwrapped
dev = env.device
DT = env.cfg.decimation * env.cfg.sim.dt
BL = env._body_length
T = int(args.seconds / DT)
nj = len(env._control_joint_ids)

# ------------------------------------------------------------------------------- policy restore
agent_yaml = Path(args.checkpoint).resolve().parents[1] / "params" / "agent.yaml"
agent_cfg = yaml.safe_load(agent_yaml.read_text())
agent_cfg["params"]["config"]["num_actors"] = 1          # player: no vecenv needed
agent_cfg["params"]["config"]["device"] = str(args.device)          # eval may use another GPU
agent_cfg["params"]["config"]["device_name"] = str(args.device)
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = str(args.checkpoint)
# the player builds its net from the env_info we inject (no rlgpu env registration needed)
agent_cfg["params"]["config"]["env_info"] = {
    "observation_space": gym.spaces.Box(-np.inf, np.inf, (env._obs_dim,)),
    "action_space": gym.spaces.Box(-1.0, 1.0, (env._num_pca,)),
    "agents": 1,
}
runner = Runner()
runner.load(agent_cfg)
player = runner.create_player()
player.restore(str(args.checkpoint))
player.has_batch_dimension = True
print(f"### policy restored from {args.checkpoint}", flush=True)


def policy_action(obs):
    ob = player._preproc_obs(obs)
    with torch.no_grad():
        res = player.model({"is_train": False, "prev_actions": None, "obs": ob,
                            "rnn_states": player.states})
    return res["mus"].detach()                            # deterministic mean


# ------------------------------------------------------------------------------- baselines
# head-first rank of each controlled joint (for the traveling-wave phase): from the calib npz
calib = np.load(env.cfg.pca_calib_path)
calib_names = [str(n) for n in calib["joint_names"]]
phi_names = [calib_names[i] for i in calib["fam1"]]      # head-first ':1' names
env_names = [env.robot.data.joint_names[i] for i in env._control_joint_ids]
rank = np.array([phi_names.index(n) for n in env_names], dtype=np.float64)  # 0=head ... 6=tail
wave_phase = torch.tensor(2 * np.pi * rank / max(1, nj - 1), device=dev, dtype=torch.float32)
wave_amp = float(np.deg2rad(args.wave_amp_deg))

# ZeF 2-PC coefficient clip at 30 Hz (frames 395:900 every 2nd), looped
basis = np.load(env.cfg.pca_basis_path)
zef_a = basis["coeffs"][395:900:2, :2].astype(np.float32)
zef_a = np.tile(zef_a, (int(np.ceil(T / len(zef_a))), 1))[:T]
zef_a_t = torch.tensor(zef_a, device=dev)

_orig_pre = env._pre_physics_step
_step_idx = {"t": 0}


def patched_pre(actions):
    t = _step_idx["t"]
    _orig_pre(actions)                                    # policy path for every env
    # env 0: frozen -- zero coefficients AND zero joint targets
    env._a[E_FROZEN] = 0.0
    env._pos_targets[E_FROZEN] = 0.0
    # env 1: scripted traveling wave. Crest at constant (omega*t - phase) moves toward
    # INCREASING phase = head->tail = forward thrust (the +phase variant measurably swam
    # backward at -0.6 BL/s: wave direction flips the thrust sign).
    q_wave = wave_amp * torch.sin(2 * np.pi * args.wave_freq * (t * DT) - wave_phase)
    lo = env._soft_joint_limits[E_WAVE, env._control_joint_ids, 0]
    hi = env._soft_joint_limits[E_WAVE, env._control_joint_ids, 1]
    env._pos_targets[E_WAVE] = torch.clamp(q_wave, lo, hi)
    env._a[E_WAVE] = 0.0                                  # a-state is meaningless for this env
    # env 2: raw ZeF 2-PC coefficients through the SAME affine map
    env._a[E_ZEF] = zef_a_t[t]
    qz = env._q0 + env._a[E_ZEF] @ env._W
    env._pos_targets[E_ZEF] = torch.clamp(
        qz, env._soft_joint_limits[E_ZEF, env._control_joint_ids, 0],
        env._soft_joint_limits[E_ZEF, env._control_joint_ids, 1])


env._pre_physics_step = patched_pre

# ------------------------------------------------------------------------------- rollout
rec = {
    "joint_pos": np.zeros((T, N_ENV, nj), np.float32),
    "pos_targets": np.zeros((T, N_ENV, nj), np.float32),
    "a": np.zeros((T, N_ENV, 2), np.float32),
    "action": np.zeros((T, N_ENV, 2), np.float32),
    "bone_pos": np.zeros((T, N_ENV, 8, 3), np.float32),
    "bone_quat": np.zeros((T, N_ENV, 8, 4), np.float32),
    "root_state": np.zeros((T, N_ENV, 13), np.float32),
    "v_body": np.zeros((T, N_ENV, 3), np.float32),
    "w_body": np.zeros((T, N_ENV, 3), np.float32),
    "reward": np.zeros((T, N_ENV), np.float32),
    "terminated": np.zeros((T, N_ENV), bool),
    "joint_vel": np.zeros((T, N_ENV, nj), np.float32),
}

obs, _ = env.reset()
# settle a moment (zero action) so the FEM relaxes before the clock starts
for _ in range(30):
    obs, *_ = env.step(torch.zeros(N_ENV, 2, device=dev))
_step_idx["t"] = 0
for t in range(T):
    act = policy_action(obs["policy"])
    _step_idx["t"] = t
    obs, rew, term, trunc, _ = env.step(act)
    d = env.robot.data
    rec["joint_pos"][t] = d.joint_pos[:, env._control_joint_ids].cpu().numpy()
    rec["joint_vel"][t] = d.joint_vel[:, env._control_joint_ids].cpu().numpy()
    rec["pos_targets"][t] = env._pos_targets.cpu().numpy()
    rec["a"][t] = env._a.cpu().numpy()
    rec["action"][t] = act.cpu().numpy()
    rec["bone_pos"][t] = d.body_link_pos_w.cpu().numpy()
    rec["bone_quat"][t] = d.body_link_quat_w.cpu().numpy()
    rec["root_state"][t] = d.root_state_w.cpu().numpy()
    rec["v_body"][t] = d.root_lin_vel_b.cpu().numpy()
    rec["w_body"][t] = d.root_ang_vel_b.cpu().numpy()
    rec["reward"][t] = rew.cpu().numpy()
    rec["terminated"][t] = term.cpu().numpy()
    if t % 150 == 0:
        vb = rec["v_body"][t, :, 0] / BL
        print(f"t={t}/{T} v_fwd_bl frozen={vb[0]:+.3f} wave={vb[1]:+.3f} zef={vb[2]:+.3f} "
              f"policy={vb[3:].mean():+.3f}", flush=True)

# commanded curvature per step per env (exact reconstruction; frozen/wave rows are not kappa-driven)
V = env._V.cpu().numpy()
kappa_cmd = basis["mean"][None, None, :] + rec["a"] @ V   # (T, E, 20)

out = Path(args.out) / Path(args.checkpoint).stem
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out / "eval_recording.npz", **rec, kappa_cmd=kappa_cmd.astype(np.float32),
    dt=DT, body_length=BL, wave_freq=args.wave_freq, wave_amp=wave_amp,
    n_policy=args.n_policy, checkpoint=str(args.checkpoint),
    a_max=env._a_max.cpu().numpy(), da_max=env._da_max.cpu().numpy(),
    link_order=calib["link_order"],
    tip_nose_bone=calib["tip_nose_bone"], tip_nose_r=calib["tip_nose_r"],
    tip_tail_bone=calib["tip_tail_bone"], tip_tail_r=calib["tip_tail_r"],
)
print(f"### saved {out}/eval_recording.npz", flush=True)
print("EVAL_ROLLOUT_DONE", flush=True)
sys.stdout.flush()
os._exit(0)
