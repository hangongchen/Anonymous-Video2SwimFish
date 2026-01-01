"""Overlay the STRIP-THEORY .obj hydro proxy on the live simfish to prove it bends/translates/rotates
WITH the fish (Training 2 deliverable).

Drives the REAL training env (SalmonSwimAMPEnv + the Misty PANELS cfg, 1 env) with a scripted
traveling-wave ACTION -- so actuation + panel hydro are exactly the training path (the fish genuinely
undulates and swims; a bare Articulation does NOT actuate this asset's motorless D6 joints). At K
frames spread across ONE tail-beat it captures the skinned .obj proxy (panel_hydro.npz vert skinning),
the live FEM body, and the bone midline, and draws a per-frame top-down overlay.

  FISH_OPEN_WATER=1 FISH_TEST_DT_HZ=120 HYDRO_TORQUE_CLIP=0.05 HYDRO_FORCE_CLIP=5.0 \
    python scripts/render_proxy_overlay.py [--frames 6] [--steps 150] [--freq 2.0] [--amp 0.8]

Output: demo_out/proxy_overlay/proxy_overlay.png (+ per-frame proxy OBJs)
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=6)
parser.add_argument("--steps", type=int, default=150)      # CONTROL steps (~30 Hz)
parser.add_argument("--freq", type=float, default=2.0)     # tail-beat Hz
parser.add_argument("--amp", type=float, default=0.8)      # action amplitude in [-1,1]
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math
import os
import sys

import gymnasium as gym
import numpy as np
import torch

DEVICE = "cuda:0"   # Isaac standalone scripts pin to cuda:0 on this box
CONTROL_HZ = 30.0
GEN = _P("${FISH_ROOT}/SimFishLib/fish_asset_pipeline/generated_usd_dataset/misty_minnow_fixed/")
NPZ = GEN + "panel_hydro.npz"
OUT_DIR = _P("${FISH_ROOT}/demo_out/proxy_overlay/")
TASK = "Template-Salmon-Swim-AMP-Misty-Panels-Direct-v0"

sys.path.insert(0, _P("${FISH_ROOT}/source/FISH"))
import FISH.tasks.direct.fish  # noqa: E402,F401  (registers the gym tasks)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

env_cfg = parse_env_cfg(TASK, device=DEVICE, num_envs=1)
env = gym.make(TASK, cfg=env_cfg)
base_env = env.unwrapped
n_act = int(base_env._num_actions)   # the exact width DirectRLEnv.step reshapes the action to
print(f"[overlay] env ready: n_act={n_act}", flush=True)

# bones come back in Isaac's body-INDEX order (tree traversal), NOT head->tail spine order -> a naive
# connecting line zig-zags. Sort by the anatomical bone number (bone1..bone9) so the midline is a
# single clean curve.
import re  # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)

_names = list(base_env.robot.data.body_names)
_bone_order = np.argsort([int(re.findall(r"\d+", n)[0]) if re.findall(r"\d+", n) else i
                          for i, n in enumerate(_names)])
print(f"[overlay] spine order: {[_names[k] for k in _bone_order]}", flush=True)

env.reset()

# ---- skinning data ----
pd = np.load(NPZ)
vert_local = torch.tensor(pd["vert_local"], device=base_env.device, dtype=torch.float32)
vert_bone = torch.tensor(pd["vert_bone"], device=base_env.device, dtype=torch.long)
M = vert_local.shape[0]
stride = max(1, M // 5000)
vl_ds, vb_ds = vert_local[::stride], vert_bone[::stride]


def skin_proxy():
    bp = base_env.robot.data.body_link_pos_w[0]
    bq = base_env.robot.data.body_link_quat_w[0]
    return (bp[vb_ds] + math_utils.quat_apply(bq[vb_ds], vl_ds)).detach().cpu().numpy()


# traveling wave HEAD->TAIL: all 3 rot-DOFs of a joint share one phase (else they fight and cancel);
# the phase ramps joint-to-joint so the body undulates as a wave instead of twisting in place.
_jidx = (torch.arange(n_act, device=base_env.device) // 3).float()
phase = _jidx / _jidx.max().clamp(min=1) * math.pi   # HALF wavelength head->tail: a C that flips -> wide tail sweep
period = max(1, int(round(CONTROL_HZ / args.freq)))
start = max(1, args.steps - period)
sample_steps = set(np.linspace(start, args.steps - 1, args.frames).astype(int).tolist())
frames = []
for s in range(args.steps):
    t = s / CONTROL_HZ
    action = (args.amp * torch.sin(2.0 * math.pi * args.freq * t + phase)).unsqueeze(0).to(base_env.device)
    env.step(action)
    if s in sample_steps:
        proxy = skin_proxy()
        sv = getattr(base_env, "_soft_view", None)
        fem = sv.get_simulation_mesh_nodal_positions()[0].detach().cpu().numpy() if sv is not None else np.zeros((0, 3))
        bones = base_env.robot.data.body_link_pos_w[0].detach().cpu().numpy()
        frames.append((s, proxy, fem, bones))
        tail = bones[np.argmin(bones[:, 0])]     # -X end = the tail (head is at +X)
        jmax = float(base_env.robot.data.joint_pos[0].abs().max())
        print(f"[overlay] step {s}: com=({bones.mean(0)[0]:.3f},{bones.mean(0)[1]:.3f},{bones.mean(0)[2]:.3f}) "
              f"tail=({tail[0]:.3f},{tail[1]:.3f},{tail[2]:.3f}) |q|max={jmax:.3f}rad", flush=True)

# ---- render ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs(OUT_DIR, exist_ok=True)
RNG = 0.20
ncol = 3
nrow = int(np.ceil(len(frames) / ncol))
fig = plt.figure(figsize=(5 * ncol, 4.4 * nrow))
for i, (s, proxy, fem, bones) in enumerate(frames):
    ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
    ctr = bones.mean(0)
    ax.scatter(proxy[:, 0], proxy[:, 1], proxy[:, 2], s=1.5, c="#5aa9e6", alpha=0.28, label=".obj proxy")
    if fem.shape[0]:
        ax.scatter(fem[:, 0], fem[:, 1], fem[:, 2], s=10, c="#e63946", alpha=0.9, label="simfish (FEM)")
    bo = bones[_bone_order]      # head->tail spine order
    ax.plot(bo[:, 0], bo[:, 1], bo[:, 2], "-o", c="k", ms=4, lw=1.8, label="bones (midline)")
    ax.view_init(elev=78, azim=-90)
    ax.set_xlim(ctr[0] - RNG, ctr[0] + RNG); ax.set_ylim(ctr[1] - RNG, ctr[1] + RNG); ax.set_zlim(ctr[2] - RNG, ctr[2] + RNG)
    ax.set_title(f"step {s}  (t={s/CONTROL_HZ:.2f}s)", fontsize=10)
    ax.set_xlabel("x (length)"); ax.set_ylabel("y (lateral)"); ax.set_zlabel("z (up)")
    if i == 0:
        ax.legend(loc="upper right", fontsize=8, markerscale=3)
    with open(OUT_DIR + f"proxy_step{s:04d}.obj", "w") as fo:
        for q in proxy:
            fo.write(f"v {q[0]:.5f} {q[1]:.5f} {q[2]:.5f}\n")

fig.suptitle("Strip-theory .obj hydro proxy (blue) tracking the live simfish (red) across one tail-beat",
             fontsize=12)
fig.tight_layout()
png = OUT_DIR + "proxy_overlay.png"
fig.savefig(png, dpi=115)
print(f"[overlay] SAVED {png}", flush=True)
print("OVERLAY_DONE", flush=True)
sys.stdout.flush()
os._exit(0)
