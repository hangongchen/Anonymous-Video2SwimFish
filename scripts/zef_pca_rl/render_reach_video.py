"""Deterministic play rollout of the reach policy -> offline mp4 (FEM point cloud + target
sequence + reach events + live reward/coefficient overlay). No Replicator/camera involved.

Usage: <env python> scripts/zef_pca_rl/render_reach_video.py --checkpoint <pth> [--seconds 40]
Output: demo_out/pca_rl_eval/reach_play_<ckpt-stem>.mp4
"""

import argparse
import sys
from pathlib import Path

REPO = Path(_P("${FISH_ROOT}"))
sys.path.insert(0, str(REPO / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seconds", type=float, default=40.0)
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

TASK = "Template-Salmon-Swim-PCA-Reach-Misty-Direct-v0"
E = 4                                             # small: runs beside training
cfg = parse_env_cfg(TASK, device=args.device, num_envs=E)
cfg.reach_frame_every_n_episodes = 10 ** 9        # in-env recorder off; we record ourselves
env = gym.make(TASK, cfg=cfg).unwrapped
dev = env.device
DT = env.cfg.decimation * env.cfg.sim.dt
T = int(args.seconds / DT)

agent_cfg = yaml.safe_load((Path(args.checkpoint).resolve().parents[1] / "params" / "agent.yaml").read_text())
agent_cfg["params"]["config"].update({"num_actors": 1, "device": str(args.device),
                                      "device_name": str(args.device)})
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = str(args.checkpoint)
agent_cfg["params"]["config"]["env_info"] = {
    "observation_space": gym.spaces.Box(-np.inf, np.inf, (env._obs_dim,)),
    "action_space": gym.spaces.Box(-1.0, 1.0, (env._num_pca,)),
    "agents": 1,
}
runner = Runner(); runner.load(agent_cfg)
player = runner.create_player(); player.restore(str(args.checkpoint))
player.has_batch_dimension = True

obs, _ = env.reset()
e0 = 0
rec = {"pts": [], "tgt": [], "reach": [], "root": [], "a": [], "dist": [], "rew": [], "nreach": []}
for t in range(T):
    ob = player._preproc_obs(obs["policy"])
    with torch.no_grad():
        act = player.model({"is_train": False, "prev_actions": None, "obs": ob,
                            "rnn_states": player.states})["mus"]
    obs, rew, term, trunc, _ = env.step(act.detach())
    rec["pts"].append(env._soft_view.get_simulation_mesh_nodal_positions()[e0].cpu().numpy())
    rec["tgt"].append(env.target_positions_w[e0].cpu().numpy().copy())
    rec["reach"].append(bool(env._reached_now[e0]))
    rec["root"].append(env.robot.data.root_state_w[e0, 0:3].cpu().numpy().copy())
    rec["a"].append(env._a[e0].cpu().numpy().copy())
    d = float(torch.linalg.norm(env.target_positions_w[e0, 0:3] - env.robot.data.root_state_w[e0, 0:3]))
    rec["dist"].append(d)
    rec["rew"].append(float(rew[e0]))
    rec["nreach"].append(int(env._targets_reached[e0]))
    if t % 300 == 0:
        print(f"t={t}/{T} dist={d:.2f} reaches={rec['nreach'][-1]}", flush=True)

import matplotlib                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


out = REPO / "demo_out/pca_rl_eval" / f"reach_play_{Path(args.checkpoint).stem}.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
pts0 = rec["pts"][0]
sc = ax.scatter(pts0[:, 0], pts0[:, 1], s=4, c="#4a90c4", alpha=0.8, label="FEM point cloud")
(trace,) = ax.plot([], [], "-", color="gray", lw=1, label="root path")
(star,) = ax.plot([], [], "*", ms=20, color="red", label="TARGET")
circ = plt.Circle((0, 0), float(env._cur_radius), fill=False, color="red", ls="--")
ax.add_patch(circ)
reach_x, reach_y = [], []
(reach_pl,) = ax.plot([], [], "o", ms=10, mfc="none", mec="green", mew=2, label="reach")
txt = ax.text(0.02, 0.99, "", transform=ax.transAxes, va="top", fontsize=9)
ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower right")
roots = np.array(rec["root"])
w = FFMpegWriter(fps=30, bitrate=2500)
with w.saving(fig, str(out), dpi=100):
    for t in range(T):
        p = rec["pts"][t]
        sc.set_offsets(p[:, :2])
        trace.set_data(roots[: t + 1, 0], roots[: t + 1, 1])
        tg = rec["tgt"][t]
        star.set_data([tg[0]], [tg[1]]); circ.center = (tg[0], tg[1])
        if rec["reach"][t]:
            reach_x.append(roots[t, 0]); reach_y.append(roots[t, 1])
            reach_pl.set_data(reach_x, reach_y)
        cx, cy = roots[t, 0], roots[t, 1]
        ax.set_xlim(cx - 1.5, cx + 1.5); ax.set_ylim(cy - 1.5, cy + 1.5)
        txt.set_text(f"t={t*DT:5.1f}s  dist={rec['dist'][t]:.2f} m  reaches={rec['nreach'][t]}\n"
                     f"a1={rec['a'][t][0]:+.1f} a2={rec['a'][t][1]:+.1f}  rew={rec['rew'][t]:+.2f}")
        w.grab_frame()
plt.close(fig)
print(f"### total reaches env0: {rec['nreach'][-1]} in {args.seconds:.0f}s")
print(f"### wrote {out}", flush=True)
print("RENDER_DONE", flush=True)
import os; sys.stdout.flush(); os._exit(0)
