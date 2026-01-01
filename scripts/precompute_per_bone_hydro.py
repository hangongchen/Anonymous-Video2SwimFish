"""Precompute PER-SLICE hydro ellipsoids for the salmon and save to npz.

The skeleton bones are pencil-thin (~2-6 mm), so hydro forces computed on them are
negligible and the fish can't push against the water. This script keeps the FEM
deformable ACTIVE once, assigns every soft-body node to its nearest bone, and fits one
ellipsoid per bone to the body SLICE around it (the real fish girth, ~0.05-0.1 m). Those
per-bone ellipsoids (semi-axes + principal rotation, in each bone's LOCAL frame) are
saved so the RL env can apply realistic, thrust-producing drag on the (deactivated-soft)
skeleton.

Output: source/FISH/FISH/tasks/direct/fish/agents/salmon/per_bone_hydro.npz
  semi (B,3)   half-extents of each bone's body slice, along its principal axes
  R_pl (B,3,3) principal->bone-local rotation
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
from pxr import UsdPhysics

sys.path.insert(0, _P("${FISH_ROOT}/source/FISH"))
from FISH.tasks.direct.fish.salmon_swim_cfg import SALMON_STAGE_PATH  # noqa: E402

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


DEVICE = "cuda:0"
DT = 1.0 / 240.0
OUT = _P("${FISH_ROOT}/source/FISH/FISH/tasks/direct/fish/agents/salmon/per_bone_hydro.npz")


def prepare_deformable_active():
    """Keep the soft body active + fast: disable internal rigid-bone colliders and
    replace the degenerate -inf collision offsets (else the GPU contact path chokes)."""
    stage = omni.usd.get_context().get_stage()
    soft_path = None
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith("/World/Fish"):
            continue
        if "/deformable_salmon" in p:
            if prim.GetAttribute("physxDeformable:simulationPoints").IsValid():
                soft_path = p
                for attr, val in [("physxCollision:contactOffset", 0.01),
                                  ("physxCollision:restOffset", 0.0),
                                  ("physxDeformable:maxDepenetrationVelocity", 1.0),
                                  ("physxDeformable:selfCollisionFilterDistance", 0.01),
                                  ("physxDeformable:enableCCD", False)]:
                    a = prim.GetAttribute(attr)
                    if a and a.IsValid():
                        a.Set(val)
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    return soft_path


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, gravity=(0.0, 0.0, 0.0), device=DEVICE))
cfg = ArticulationCfg(
    prim_path="/World/Fish/skeleton",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SALMON_STAGE_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16,
            solver_velocity_iteration_count=2),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True)),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=25.0, damping=20.0)})
robot = Articulation(cfg)
soft_path = prepare_deformable_active()

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
print(f"[precompute] {B} bones, {M} soft nodes", flush=True)

# assign each node to the nearest bone
d = torch.cdist(N, P)                  # (M,B)
nearest = d.argmin(dim=1)              # (M,)

semi = torch.zeros((B, 3), device=DEVICE)
R_pl = torch.eye(3, device=DEVICE).unsqueeze(0).repeat(B, 1, 1)
counts = []
for b in range(B):
    nb = N[nearest == b]
    counts.append(int(nb.shape[0]))
    if nb.shape[0] < 4:
        continue
    # into the bone's LOCAL frame
    nl = quat_rotate_inverse(Q[b:b + 1].expand(nb.shape[0], -1), nb - P[b])
    nl = nl - nl.mean(0)
    _, _, Vh = torch.linalg.svd(nl, full_matrices=False)     # Vh rows = principal axes (bone-local)
    proj = nl @ Vh.transpose(0, 1)
    # robust half-extent: 95th percentile of |proj| so a stray node doesn't blow it up
    s = torch.quantile(proj.abs(), 0.95, dim=0).clamp_min(2e-3)
    semi[b] = s
    R_pl[b] = Vh.transpose(0, 1)       # principal -> bone-local (columns = principal axes)

# fill empty/sparse bones with the median of the populated ones
pop = semi.sum(-1) > 0
if pop.any():
    med = semi[pop].median(0).values
    semi[~pop] = med
print(f"[precompute] node counts per bone: {counts}", flush=True)
print(f"[precompute] semi-axes (m): mean={[round(float(x),4) for x in semi.mean(0)]}  "
      f"body 2*thick(min axis mean)={float(2*semi.min(-1).values.mean()):.4f}  "
      f"(vs thin-bone ~0.0065)", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, semi=semi.cpu().numpy(), R_pl=R_pl.cpu().numpy(),
         counts=np.array(counts), num_bones=B)
print(f"[precompute] saved {OUT}", flush=True)
print("PRECOMPUTE_DONE", flush=True)
sys.stdout.flush()
os._exit(0)
