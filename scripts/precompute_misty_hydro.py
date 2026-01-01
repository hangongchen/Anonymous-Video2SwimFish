"""Precompute PER-SLICE hydro ellipsoids for the AUTO-SKELETON fish and save to npz.

Same method as precompute_per_bone_hydro.py (salmon): spawn the fish with its FEM body active,
assign each soft node to its nearest bone, fit one ellipsoid per bone to the body slice around it,
save (semi, R_pl) in each bone's LOCAL frame. The AMP env then applies realistic thrust-producing
drag instead of the thin-bone inertia fallback ("weak thrust").

Differences from the salmon script: the SimFishLib asset needs the multi-articulation-root and
reflected-deformable fixups + the `final_mesh` token — all reused from the env's own
_prepare_env_assets so the setup is identical to training.

Output: SimFishLib/fish_asset_pipeline/generated_usd_dataset/misty_minnow_fixed/per_bone_hydro.npz
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import sys

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.math import quat_rotate_inverse

import omni.usd

sys.path.insert(0, _P("${FISH_ROOT}/source/FISH"))
from FISH.tasks.direct.fish.salmon_swim_env import _prepare_env_assets  # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


DEVICE = "cuda:0"
DT = 1.0 / 240.0
USD = _P("${FISH_ROOT}/SimFishLib/fish_asset_pipeline/generated_usd_dataset/"
       "misty_minnow_fixed/fish_articulated.usd")
OUT = _P("${FISH_ROOT}/SimFishLib/fish_asset_pipeline/generated_usd_dataset/"
       "misty_minnow_fixed/per_bone_hydro.npz")
TOKEN = "mesh_001"

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, gravity=(0.0, 0.0, 0.0), device=DEVICE))
cfg = ArticulationCfg(
    prim_path="/World/Fish/skeleton",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16,
            solver_velocity_iteration_count=2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=25.0, damping=20.0)})
robot = Articulation(cfg)

# identical asset fixups to training (root fixup, FEM material, reflected xform, offsets)
stage = omni.usd.get_context().get_stage()
_prepare_env_assets(stage, "/World/Fish/skeleton", control_mode="position", keep_deformable=True,
                    deformable_token=TOKEN, fix_multi_articulation_root=True,
                    fix_reflected_deformable=True, keep_bone_colliders=False,
                    fem_youngs_modulus=1.0e5, fem_elasticity_damping=0.05, fem_damping_scale=1.0)
soft_path = None
for prim in stage.Traverse():
    p = str(prim.GetPath())
    if p.startswith("/World/Fish") and TOKEN in p and \
            prim.GetAttribute("physxDeformable:simulationPoints").IsValid():
        soft_path = p
        break
assert soft_path is not None, "deformable prim not found"

sim.reset()
for _ in range(3):
    robot.write_data_to_sim(); sim.step(render=False); robot.update(DT)

from isaacsim.core.prims import DeformablePrim  # noqa: E402
soft_view = DeformablePrim(prim_paths_expr=soft_path, reset_xform_properties=False)
soft_view.initialize()

P = robot.data.body_link_pos_w[0].to(DEVICE).float()        # (B,3) bone world positions
Q = robot.data.body_link_quat_w[0].to(DEVICE).float()       # (B,4) bone world quats (wxyz)
N = soft_view.get_simulation_mesh_nodal_positions()[0].to(DEVICE).float()   # (M,3) node world positions
B = P.shape[0]
M = N.shape[0]
print(f"[precompute-autoskel] {B} bones, {M} soft nodes", flush=True)

d = torch.cdist(N, P)
nearest = d.argmin(dim=1)

semi = torch.zeros((B, 3), device=DEVICE)
R_pl = torch.eye(3, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)
counts = []
for b in range(B):
    nb = N[nearest == b]
    counts.append(int(nb.shape[0]))
    if nb.shape[0] < 4:
        continue
    nl = quat_rotate_inverse(Q[b:b + 1].expand(nb.shape[0], -1), nb - P[b])
    nl = nl - nl.mean(0)
    _, _, Vh = torch.linalg.svd(nl, full_matrices=False)
    proj = nl @ Vh.transpose(0, 1)
    s = torch.quantile(proj.abs(), 0.95, dim=0).clamp_min(2e-3)
    semi[b] = s
    R_pl[b] = Vh.transpose(0, 1)

pop = semi.sum(-1) > 0
if pop.any():
    med = semi[pop].median(0).values
    semi[~pop] = med
print(f"[precompute-autoskel] node counts per bone: {counts}", flush=True)
print(f"[precompute-autoskel] semi-axes (m): mean={[round(float(x),4) for x in semi.mean(0)]}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, semi=semi.cpu().numpy(), R_pl=R_pl.cpu().numpy(),
         counts=np.array(counts), num_bones=B)
print(f"[precompute-autoskel] saved {OUT}", flush=True)
print("PRECOMPUTE_DONE", flush=True)
sys.stdout.flush()
os._exit(0)
