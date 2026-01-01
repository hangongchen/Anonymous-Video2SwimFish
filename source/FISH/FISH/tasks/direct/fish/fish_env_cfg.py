# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass


ASSETS_DIR = Path(__file__).resolve().parent / "agents"
# Use the articulated fish asset (copy).
FISH_STAGE_PATH = ASSETS_DIR / "hollowFish.usd"

# Ground plane / environment layout constants.
# Keep default ground/spacing and scale the asset to meters.
# Shrink the ground plane by 100x for debugging.
GROUND_AREA_SCALE = math.sqrt(1000.0) / 500
_DEFAULT_GROUND_SIZE = GroundPlaneCfg().size
FISH_GROUND_SIZE = tuple(dim * GROUND_AREA_SCALE for dim in _DEFAULT_GROUND_SIZE)
# Add a small padding (5%) so replicated environments do not overlap.
FISH_ENV_SPACING = max(FISH_GROUND_SIZE) * 1.05


FISH_ARTICULATION_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/skeleton",
    articulation_root_prim_path=None,
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(FISH_STAGE_PATH),
        # Authored asset uses centimeters; convert to meters for a consistent world scale.
        # scale=(1.0, 1.0, 1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=5.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    # Orient fish with head along -X and back aligned with +Z (level with ground).
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0),
        rot=(0.5, 0.5, -0.5, -0.5),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1e5,
            stiffness=2000.0,
            damping=200.0,
        )
    },
)


TARGET_MARKER_CFG = sim_utils.CuboidCfg(
    # Marker size in meters.
    size=(0.2, 0.2, 0.2),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
)


@configclass
class FishEnvCfg(DirectRLEnvCfg):
    # env timing
    decimation = 8
    # 100 env steps at 1/240s with decimation=8 => 100 * (8/240) = 3.333... seconds
    episode_length_s = 2000 * (decimation / 240)
    # placeholder spaces (actual dims resolved at runtime)
    action_space = 1
    observation_space = 1
    state_space = 0

    # simulation settings (GPU by default via device="cuda:0")
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = FISH_ARTICULATION_CFG

    # scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=FISH_ENV_SPACING,
        replicate_physics=False,  # Fish USD contains attachments that PhysX replication cannot duplicate.
    )

    # action scaling / debug
    action_scale = 30  # was 8000
    debug_logs = False

    @configclass
    class ObservationScalesCfg:
        joint_pos = 1.0
        joint_vel = 0.05
        root_lin_vel = 0.2
        root_ang_vel = 0.2

    obs_scales: ObservationScalesCfg = ObservationScalesCfg()

    @configclass
    class RewardScalesCfg:
        distance = 10
        sparse_progress = 0.0
        heading = 0.00
        orientation = 0.00
        success = 200
        time = 0.001

    rew_scales: RewardScalesCfg = RewardScalesCfg()

    # Height thresholds in meters.
    target_root_height = 0.3
    termination_height = -1.0
    joint_limit_margin = 0.2

    initial_root_pos_range = (-0.0, 0.0)
    initial_root_rot_range = (-math.pi, math.pi)
    initial_joint_pos_range = (-0, 0)
    initial_joint_vel_range = (-0, 0)
    # Target offset in meters.
    target_offset_xy = (0.0, 0.0)
    target_height = 0.05
    target_marker_height = 0.05  # visible in meter units
    target_marker_cfg: sim_utils.CuboidCfg = TARGET_MARKER_CFG
    # Episode success when distance to target (XY) is within this radius.
    success_radius = 1.0
    # Progress milestone spacing for sparse reward (fraction of start distance).
    sparse_progress_step = 0.25
