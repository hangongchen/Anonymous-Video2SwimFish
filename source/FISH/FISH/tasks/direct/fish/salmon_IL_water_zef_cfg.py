# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test config: run the analytic-water IMITATION task with the SimFishLib zebrafish.

Same IL/analytic-water pipeline as `SalmonILWaterEnvCfg`, but:
  * the AGENT is swapped from the salmon to the SimFishLib video->USD zebrafish
    (`fish_articulated.usd`, 7 bones / 6 D6 joints, deformable mesh named `final_mesh`),
  * the reference is swapped from the 451-frame real-salmon point cloud to a real zebrafish
    body-shape / undulation sequence carved FROM the 3D-ZeF VIDEO (segment -> two-view silhouette
    carve -> per-frame point cloud; scripts/build_zef_video_pointclouds.py), consumed in the
    canonical-frame `real_pointcloud` chamfer mode,
  * per-bone hydro is left to the inertia fallback (the salmon 10-bone npz doesn't fit
    the 7-bone zebrafish),
  * `deformable_prim_token` is `final_mesh` so the env's deformable scans find the FEM body,
  * Early Termination is disabled and the env count is tiny so the run actually exercises the
    physics long enough to see whether the (uncooked, placeholder-physics) zebrafish blows up.
"""

from __future__ import annotations

import copy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .salmon_IL_water_cfg import SalmonILWaterEnvCfg
from .salmon_swim_cfg import SALMON_SWIM_ARTICULATION_CFG

_ZEF_DIR = (
    Path(__file__).resolve().parents[6]
    / "SimFishLib" / "fish_asset_pipeline" / "generated_usd_dataset"
    / "vlm_rigged_single_fish" / "zef_f0001"
)
# prefer the sim-ready COOKED asset (proper FEM sim mesh via cook_deformable_isaac.py) if present,
# else fall back to the raw pipeline output.
_ZEF_USD = _ZEF_DIR / "fish_articulated_cooked.usd"
if not _ZEF_USD.exists():
    _ZEF_USD = _ZEF_DIR / "fish_articulated.usd"

# reuse the salmon articulation cfg (same solver/rigid/actuator setup, gentle stable gains) but
# point it at the zebrafish USD and give it a level identity heading (its body long axis is +X).
_ZEF_ARTICULATION_CFG: ArticulationCfg = copy.deepcopy(SALMON_SWIM_ARTICULATION_CFG)
_ZEF_ARTICULATION_CFG.spawn.usd_path = str(_ZEF_USD)
_ZEF_ARTICULATION_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.0),
    rot=(1.0, 0.0, 0.0, 0.0),
)


@configclass
class SalmonILWaterZefEnvCfg(SalmonILWaterEnvCfg):
    robot_cfg: ArticulationCfg = _ZEF_ARTICULATION_CFG

    # the zebrafish deformable mesh is named `final_mesh`, not `deformable_salmon`
    deformable_prim_token = "final_mesh"

    # the SimFishLib pipeline stamps ArticulationRootAPI on every bone -> Isaac sees 7 roots and
    # errors; collapse to a single root on the spawn prim.
    fix_multi_articulation_root = True

    # the zebrafish deformable mesh has a reflected (negative-scale) xform -> right-hand it so the
    # FEM DeformablePrim view can initialize.
    fix_reflected_deformable = True

    # the salmon per-slice hydro npz is keyed to 10 bones; the zebrafish has 7 -> use the
    # inertia-derived thin-bone fallback (weak thrust, but shape-correct for 7 bodies).
    per_bone_hydro_path = ""

    # reference = the REAL zebrafish body shape / undulation, carved FROM THE VIDEO frame-by-frame
    # (SimFishLib two-view silhouette carve -> 64^3 voxel -> point cloud). Consumed in the
    # canonical-frame `real_pointcloud` chamfer mode (SVD-aligned gait imitation), NOT demo mode:
    # the carve canonicalizes each frame (head->+X, PCA-upright) and discards world position, so
    # there is no world trajectory to match -- only the deforming body shape.
    demo_path = ""    # disable world-frame demo mode
    real_pointcloud_dir = str(
        Path(__file__).resolve().parents[6] / "demo_out" / "zef_video" / "pointclouds"
    )
    real_pointcloud_glob = "zefvid_*.npy"

    # tiny env count: enough to exercise cloning + the O(num_envs) chamfer loop, cheap to cook.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=8.0,
        replicate_physics=False,
    )

    # per-step blow-up / throughput logging on
    debug_print = True
