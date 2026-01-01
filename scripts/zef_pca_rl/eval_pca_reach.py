"""Deterministic evaluation of a PCA reach policy (v1 plain or v2 head-first) with the
anti-gliding diagnostics + video. Run once per checkpoint; compare the metrics.json files.

  <env python> scripts/zef_pca_rl/eval_pca_reach.py \
      --task Template-Salmon-Swim-PCA-Reach-Misty-Direct-v0 \
      --checkpoint <pth> --tag old_v1 [--episodes 2] [--n_envs 16]

Outputs demo_out/pca_reach_eval/<tag>/: metrics.json, recording npz, video env0.mp4, plots.
Metrics: success rate (reached targets / spawned targets), reaches/episode, time-to-reach,
mean initial + final target distance, forward/lateral body-frame speed, target-directed
velocity, yaw rate, FINAL-APPROACH (dist<0.5 m) heading error/cos + lateral speed, PCA
coefficient stats, FEM blow-ups.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(_P("${FISH_ROOT}"))
sys.path.insert(0, str(REPO / "source"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-Salmon-Swim-PCA-Reach-Misty-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--n_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=None,
                    help="env seed: same seed => identical spawn states + identical FIRST targets "
                         "across methods (later targets are policy-conditioned by construction)")
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
if args.seed is not None:
    cfg.seed = int(args.seed)
env = gym.make(args.task, cfg=cfg).unwrapped
dev = env.device
DT = env.cfg.decimation * env.cfg.sim.dt
BL = env._body_length
T = int(args.episodes * env.max_episode_length)

agent_cfg = yaml.safe_load((Path(args.checkpoint).resolve().parents[1] / "params" / "agent.yaml").read_text())
agent_cfg["params"]["config"].update({"num_actors": 1, "device": str(args.device),
                                      "device_name": str(args.device)})
agent_cfg["params"]["load_checkpoint"] = True
agent_cfg["params"]["load_path"] = str(args.checkpoint)
agent_cfg["params"]["config"]["env_info"] = {
    "observation_space": gym.spaces.Box(-np.inf, np.inf, (env._obs_dim,)),
    "action_space": gym.spaces.Box(-1.0, 1.0, env.single_action_space.shape),  # PCA=2 / CPG=3
    "agents": 1,
}
runner = Runner(); runner.load(agent_cfg)
player = runner.create_player(); player.restore(str(args.checkpoint))
player.has_batch_dimension = True
print(f"### restored {args.checkpoint}", flush=True)

obs, _ = env.reset()
R = {k: [] for k in ("root", "quat", "vb", "wb", "tgt", "dist", "reach", "a", "nreach",
                     "term", "pts0", "hcos", "vtgt")}
tstart = np.zeros(E, int)        # step each env's current target was spawned
ttr = []                          # time-to-reach (s) per reach event
d0s = []                          # initial distance of each target at spawn
final_ds = []                     # distance at episode end (unfinished target)
prev_nreach = np.zeros(E)
for t in range(T):
    ob = player._preproc_obs(obs["policy"])
    with torch.no_grad():
        act = player.model({"is_train": False, "prev_actions": None, "obs": ob,
                            "rnn_states": player.states})["mus"]
    obs, rew, term, trunc, _ = env.step(act.detach())
    d = env.robot.data
    root = d.root_state_w
    delta = env.target_positions_w[:, 0:3] - root[:, 0:3]
    dist = delta.norm(dim=1)
    tdir = delta / dist.clamp(min=1e-6).unsqueeze(-1)
    nose = math_utils.quat_apply(root[:, 3:7],
                                 torch.tensor([1.0, 0, 0], device=dev).repeat(E, 1))
    nose = nose / nose.norm(dim=1, keepdim=True).clamp(min=1e-6)
    hcos = (nose * tdir).sum(-1)
    vtgt = (root[:, 7:10] * tdir).sum(-1) / BL
    reached = env._reached_now.cpu().numpy().astype(bool)
    for e in np.nonzero(reached)[0]:
        ttr.append((t - tstart[e]) * DT)
        tstart[e] = t
        d0s.append(float(dist[e]))                    # NEW target's spawn distance
    done = (trunc | term).cpu().numpy().astype(bool)
    for e in np.nonzero(done)[0]:
        final_ds.append(float(dist[e]))
        tstart[e] = t
    R["root"].append(root[:, 0:3].cpu().numpy()); R["quat"].append(root[:, 3:7].cpu().numpy())
    R["vb"].append(d.root_lin_vel_b.cpu().numpy()); R["wb"].append(d.root_ang_vel_b.cpu().numpy())
    R["tgt"].append(env.target_positions_w.cpu().numpy().copy())
    R["dist"].append(dist.cpu().numpy()); R["reach"].append(reached.copy())
    R["a"].append(env._a.cpu().numpy().copy())
    R["nreach"].append(env._targets_reached.cpu().numpy().copy())
    R["term"].append(term.cpu().numpy()); R["hcos"].append(hcos.cpu().numpy())
    R["vtgt"].append(vtgt.cpu().numpy())
    R["pts0"].append(env._soft_view.get_simulation_mesh_nodal_positions()[0].cpu().numpy())
    if t % 300 == 0:
        print(f"t={t}/{T} dist0={float(dist[0]):.2f} reaches0={int(env._targets_reached[0])}",
              flush=True)

A = {k: np.array(v) for k, v in R.items()}
total_reaches = len(ttr)
spawned = total_reaches + len(final_ds)               # every unfinished target counts once
fin = A["dist"] < 0.5                                 # final-approach mask (T,E)
herr = np.rad2deg(np.arccos(np.clip(A["hcos"], -1, 1)))
metrics = {
    "checkpoint": str(args.checkpoint), "task": args.task, "episodes": args.episodes,
    "n_envs": E, "seconds": T * DT,
    "success_rate_targets": total_reaches / max(spawned, 1),
    "reaches_total": total_reaches,
    "reaches_per_episode": total_reaches / (args.episodes * E),
    "time_to_reach_mean_s": float(np.mean(ttr)) if ttr else None,
    "time_to_reach_median_s": float(np.median(ttr)) if ttr else None,
    "initial_target_dist_mean": float(np.mean(d0s)) if d0s else None,
    "final_dist_unreached_mean": float(np.mean(final_ds)) if final_ds else None,
    "v_fwd_bl_mean": float(A["vb"][:, :, 0].mean() / BL),
    "v_lat_bl_rms": float(np.sqrt((np.linalg.norm(A["vb"][:, :, 1:3], axis=-1) ** 2).mean()) / BL),
    "v_target_bl_mean": float(A["vtgt"].mean()),
    "yaw_rate_abs_mean": float(np.abs(A["wb"][:, :, 1]).mean()),
    "heading_cos_mean": float(A["hcos"].mean()),
    "final_approach": {
        "heading_err_deg_mean": float(herr[fin].mean()) if fin.any() else None,
        "heading_cos_mean": float(A["hcos"][fin].mean()) if fin.any() else None,
        "v_lat_bl_rms": float(np.sqrt((np.linalg.norm(A["vb"][:, :, 1:3], axis=-1)[fin] ** 2).mean()) / BL)
        if fin.any() else None,
        "v_target_bl_mean": float(A["vtgt"][fin].mean()) if fin.any() else None,
        "n_steps": int(fin.sum()),
    },
    "a1_rms": float(np.sqrt((A["a"][:, :, 0] ** 2).mean())),
    "a2_rms": float(np.sqrt((A["a"][:, :, 1] ** 2).mean())),
    "fem_blowups": int(A["term"].sum()),
}
out = REPO / "demo_out/pca_reach_eval" / args.tag
out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
np.savez_compressed(out / "recording.npz", **A, dt=DT, body_length=BL)
print(json.dumps(metrics, indent=2), flush=True)

# ---- env0 video: point cloud + nose arrow + target + reach events ----
import matplotlib                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
sc = ax.scatter(A["pts0"][0][:, 0], A["pts0"][0][:, 1], s=4, c="#4a90c4", alpha=0.8)
(trace,) = ax.plot([], [], "-", color="gray", lw=1)
(star,) = ax.plot([], [], "*", ms=20, color="red")
circ = plt.Circle((0, 0), 0.2, fill=False, color="red", ls="--"); ax.add_patch(circ)
arr = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="black", lw=2))
(reach_pl,) = ax.plot([], [], "o", ms=10, mfc="none", mec="green", mew=2)
txt = ax.text(0.02, 0.99, "", transform=ax.transAxes, va="top", fontsize=9)
ax.set_aspect("equal")
rx, ry = [], []
w = FFMpegWriter(fps=30, bitrate=2500)
with w.saving(fig, str(out / "env0.mp4"), dpi=100):
    for t in range(T):
        sc.set_offsets(A["pts0"][t][:, :2])
        trace.set_data(A["root"][: t + 1, 0, 0], A["root"][: t + 1, 0, 1])
        tg = A["tgt"][t, 0]
        star.set_data([tg[0]], [tg[1]]); circ.center = (tg[0], tg[1])
        p = A["root"][t, 0]
        q = A["quat"][t, 0]                          # wxyz; nose = R(q) @ +x, world XY components
        nose = (1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]))
        arr.xy = (p[0] + 0.25 * nose[0], p[1] + 0.25 * nose[1]); arr.set_position((p[0], p[1]))
        if A["reach"][t, 0]:
            rx.append(p[0]); ry.append(p[1]); reach_pl.set_data(rx, ry)
        ax.set_xlim(p[0] - 1.5, p[0] + 1.5); ax.set_ylim(p[1] - 1.5, p[1] + 1.5)
        txt.set_text(f"t={t*DT:5.1f}s dist={A['dist'][t,0]:.2f} m "
                     f"hcos={A['hcos'][t,0]:+.2f} reaches={int(A['nreach'][t,0])}\n"
                     f"a1={A['a'][t,0,0]:+.1f} a2={A['a'][t,0,1]:+.1f}  (arrow = nose)")
        w.grab_frame()
plt.close(fig)
print(f"### wrote {out}/env0.mp4", flush=True)
print("REACH_EVAL_DONE", flush=True)
sys.stdout.flush()
import os
os._exit(0)
