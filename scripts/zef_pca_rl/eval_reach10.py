"""Deterministic 10-target evaluation on SEEDED identical target sequences.

Run once per (method, seed); both methods MUST use the same --seed so they face identical
per-(env,attempt) relative target draws (demo_out/reach10_eval/target_seq_seed<k>.npz).

  <env python> scripts/zef_pca_rl/eval_reach10.py \
      --task Template-Salmon-Reach10-PCA-Misty-Direct-v0 --checkpoint <pth> \
      --tag pca_seed101 --seed 101

Outputs demo_out/reach10_eval/<tag>/: metrics.json (incl. the per-episode 0/1 target tables),
recording.npz, env0.mp4 (target number + cumulative count shown in the VIDEO ONLY -- the
policy observation never contains them).
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(_P("${FISH_ROOT}"))
sys.path.insert(0, str(REPO / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--n_envs", type=int, default=16)
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
import isaaclab.utils.math as math_utils        # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


E = args.n_envs
cfg = parse_env_cfg(args.task, device=args.device, num_envs=E)
cfg.reach_frame_every_n_episodes = 10 ** 9
cfg.seed = int(args.seed)
cfg.target_seq_path = str(REPO / f"demo_out/reach10_eval/target_seq_seed{args.seed:03d}.npz")
env = gym.make(args.task, cfg=cfg).unwrapped
dev = env.device
DT = env.cfg.decimation * env.cfg.sim.dt
BL = env._body_length
N_ATT = env._n_att
T_MAX = N_ATT * env._att_timeout_steps + 120

agent_cfg = yaml.safe_load((Path(args.checkpoint).resolve().parents[1] / "params" / "agent.yaml").read_text())
agent_cfg["params"]["config"].update({"num_actors": 1, "device": str(args.device),
                                      "device_name": str(args.device)})
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = str(args.checkpoint)
agent_cfg["params"]["config"]["env_info"] = {
    "observation_space": gym.spaces.Box(-np.inf, np.inf, (env._obs_dim,)),
    "action_space": gym.spaces.Box(-1.0, 1.0, env.single_action_space.shape),
    "agents": 1,
}
runner = Runner(); runner.load(agent_cfg)
player = runner.create_player(); player.restore(str(args.checkpoint))
player.has_batch_dimension = True
print(f"### restored {args.checkpoint}", flush=True)

obs, _ = env.reset()
done_ep = np.zeros(E, dtype=bool)
att_start = np.zeros(E, dtype=int)
ttr = []                                  # (env, attempt, seconds) successful reaches only
rec = {k: [] for k in ("root", "quat", "vb", "wb", "tgt", "dist", "reach", "att", "nreach",
                       "term", "pts0", "hcos")}
outcomes = [None] * E
t = 0
while t < T_MAX and not done_ep.all():
    ob = player._preproc_obs(obs["policy"])
    with torch.no_grad():
        act = player.model({"is_train": False, "prev_actions": None, "obs": ob,
                            "rnn_states": player.states})["mus"]
    prev_att = env._att_idx.cpu().numpy().copy()
    obs, rew, term, trunc, _ = env.step(act.detach())
    d = env.robot.data
    root = d.root_state_w
    delta = env.target_positions_w[:, 0:3] - root[:, 0:3]
    dist = delta.norm(dim=1)
    tdir = delta / dist.clamp(min=1e-6).unsqueeze(-1)
    nose = math_utils.quat_apply(root[:, 3:7], torch.tensor([1.0, 0, 0], device=dev).repeat(E, 1))
    hcos = (nose / nose.norm(dim=1, keepdim=True).clamp(min=1e-6) * tdir).sum(-1)
    reached = env.extras["success"].cpu().numpy().astype(bool)
    now_att = env._att_idx.cpu().numpy()
    for e in np.nonzero(now_att > prev_att)[0]:               # an attempt just ended
        if reached[e]:
            ttr.append((int(e), int(prev_att[e]), (t - att_start[e]) * DT))
        att_start[e] = t + 1
    ended = (term | trunc).cpu().numpy().astype(bool)
    for e in np.nonzero(ended & ~done_ep)[0]:
        if len(env._ep_outcomes) > 0:
            outcomes[e] = None                                # filled from deque after loop
        done_ep[e] = True
        att_start[e] = t + 1
    rec["root"].append(root[:, 0:3].cpu().numpy()); rec["quat"].append(root[:, 3:7].cpu().numpy())
    rec["vb"].append(d.root_lin_vel_b.cpu().numpy()); rec["wb"].append(d.root_ang_vel_b.cpu().numpy())
    rec["tgt"].append(env.target_positions_w.cpu().numpy().copy())
    rec["dist"].append(dist.cpu().numpy()); rec["reach"].append(reached.copy())
    rec["att"].append(now_att.copy()); rec["nreach"].append(env._targets_reached.cpu().numpy().copy())
    rec["term"].append(term.cpu().numpy())
    rec["hcos"].append(hcos.cpu().numpy())
    rec["pts0"].append(env._soft_view.get_simulation_mesh_nodal_positions()[0].cpu().numpy())
    if t % 600 == 0:
        print(f"t={t} done={int(done_ep.sum())}/{E} att0={int(now_att[0])} "
              f"reached0={int(env._targets_reached[0])}", flush=True)
    t += 1

# per-episode outcome tables: first E vectors pushed to the deque during this rollout
tables = [v.tolist() for v in list(env._ep_outcomes)[-int(done_ep.sum()):]]
A = {k: np.array(v) for k, v in rec.items()}
total = sum(sum(x) for x in tables)
n_ep = len(tables)
fin = A["dist"] < 0.5
metrics = {
    "checkpoint": str(args.checkpoint), "task": args.task, "seed": args.seed,
    "n_episodes": n_ep, "n_targets_per_ep": N_ATT,
    "target_tables": tables,
    "targets_reached_per_ep": [int(sum(x)) for x in tables],
    "overall_success_rate": total / max(1, n_ep * N_ATT),
    "per_target_success_rate": [float(np.mean([x[i] for x in tables])) for i in range(N_ATT)] if tables else [],
    "time_to_reach_mean_s": float(np.mean([x[2] for x in ttr])) if ttr else None,
    "time_to_reach_median_s": float(np.median([x[2] for x in ttr])) if ttr else None,
    "v_fwd_bl_mean": float(A["vb"][:, :, 0].mean() / BL),
    "v_lat_bl_rms": float(np.sqrt((np.linalg.norm(A["vb"][:, :, 1:3], axis=-1) ** 2).mean()) / BL),
    "yaw_rate_abs_mean": float(np.abs(A["wb"][:, :, 1]).mean()),
    "fem_blowups": int(A["term"].sum()),
}
hcA = np.array(rec["hcos"])
metrics["heading_cos_mean"] = float(hcA.mean())
metrics["final_approach_heading_err_deg"] = float(
    np.rad2deg(np.arccos(np.clip(hcA[fin], -1, 1))).mean()) if fin.any() else None
out = REPO / "demo_out/reach10_eval" / args.tag
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
np.savez_compressed(out / "recording.npz", **A, dt=DT, body_length=BL)
print(json.dumps({k: v for k, v in metrics.items() if k != "target_tables"}, indent=2))
print("tables:", tables, flush=True)

# ---- env0 video: target NUMBER + cumulative count in the OVERLAY only ----
import matplotlib                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
T = A["root"].shape[0]
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
sc = ax.scatter(A["pts0"][0][:, 0], A["pts0"][0][:, 1], s=4, c="#4a90c4", alpha=0.8)
(tr,) = ax.plot([], [], "-", color="gray", lw=1)
(st,) = ax.plot([], [], "*", ms=20, color="red")
ci = plt.Circle((0, 0), 0.2, fill=False, color="red", ls="--"); ax.add_patch(ci)
ar = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=2))
(rp,) = ax.plot([], [], "o", ms=10, mfc="none", mec="green", mew=2)
tx = ax.text(0.02, 0.99, "", transform=ax.transAxes, va="top", fontsize=10)
ax.set_aspect("equal")
rx, ry = [], []
w = FFMpegWriter(fps=30, bitrate=2500)
with w.saving(fig, str(out / "env0.mp4"), dpi=100):
    for t in range(T):
        sc.set_offsets(A["pts0"][t][:, :2])
        tr.set_data(A["root"][: t + 1, 0, 0], A["root"][: t + 1, 0, 1])
        tg = A["tgt"][t, 0]
        st.set_data([tg[0]], [tg[1]]); ci.center = (tg[0], tg[1])
        p, q = A["root"][t, 0], A["quat"][t, 0]
        nose = (1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]))
        ar.xy = (p[0] + 0.25 * nose[0], p[1] + 0.25 * nose[1]); ar.set_position((p[0], p[1]))
        if A["reach"][t, 0]:
            rx.append(p[0]); ry.append(p[1]); rp.set_data(rx, ry)
        ax.set_xlim(p[0] - 1.5, p[0] + 1.5); ax.set_ylim(p[1] - 1.5, p[1] + 1.5)
        tx.set_text(f"TARGET #{min(int(A['att'][t,0])+1, N_ATT)}/10   "
                    f"reached {int(A['nreach'][t,0])}   dist={A['dist'][t,0]:.2f} m   "
                    f"t={t*DT:5.1f}s")
        w.grab_frame()
plt.close(fig)
print(f"### wrote {out}/env0.mp4")
print("REACH10_EVAL_DONE", flush=True)
sys.stdout.flush()
import os
os._exit(0)
