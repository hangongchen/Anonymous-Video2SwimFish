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
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


ASSETS_DIR = Path(__file__).resolve().parent / "agents"
SALMON_STAGE_PATH = ASSETS_DIR / "salmon" / "uniformed_scale_small_salmon.usd"
WATER_ENV_STAGE_PATH = ASSETS_DIR / "fluid_env" / "Water_env.usd"
DEFAULT_REAL_POINTCLOUD_DIR = Path(__file__).resolve().parents[6] / "pointclouds"

# The imported Water_env scene already contains the floor and side walls.
WATER_ENV_SPACING = 8.0


SALMON_IL_ARTICULATION_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/skeleton",
    articulation_root_prim_path=None,
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SALMON_STAGE_PATH),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        # Same heading as before, but rolled 180 degrees around the fish body axis
        # so the Salmon IL experiment starts upside down.
        rot=(0.5, -0.5, 0.5, -0.5),
    ),
    actuators={
        "all_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=1e5,
            stiffness=25.0,
            damping=20.0,
        )
    },
)

@configclass
class SalmonILEnvCfg(DirectRLEnvCfg):
    decimation = 8
    episode_length_s = 2000 * (decimation / 240)
    action_space = 1
    observation_space = 1
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        physx=PhysxCfg(
            enable_external_forces_every_iteration=True,
            gpu_collision_stack_size=2**30,
            gpu_heap_capacity=2**30,
            gpu_temp_buffer_capacity=2**28,
            gpu_max_soft_body_contacts=2**22,
            gpu_max_particle_contacts=2**28,
        ),
    )

    robot_cfg: ArticulationCfg = SALMON_IL_ARTICULATION_CFG

    # The Chamfer imitation pipeline operates on point clouds every step, so keep
    # the default env count smaller than the locomotion task.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=WATER_ENV_SPACING,
        replicate_physics=False,
    )

    # Keep Salmon IL closer to the post-NaN fish task control magnitude.
    # The original fish fix came from stabilizing the dynamics/reward path,
    # not from making the action clamp do the work.
    action_scale = 5.0
    debug_logs = True
    # Water_env's sampled fluid volume starts around z=1.5, so keep the small
    # salmon comfortably inside it instead of intersecting the lower boundary.
    asset_spawn_height_offset = 1.5
    preserve_articulation_root_pose_on_reset = True
    pointcloud_capture_enabled = False
    pointcloud_capture_interval_s = 10.0
    pointcloud_capture_env_id = 0
    pointcloud_capture_dir: str | None = None
    # The authored Water_env particle system still crashes the deformable
    # salmon's GPU contact path on this machine, so keep the stable backdrop
    # mode as the default and only re-enable particles for focused debugging.
    water_particles_enabled = False
    water_particle_max_samples = 8_000
    water_particle_sampling_distance = 0.2
    disable_internal_rigid_collisions = True
    alignment_overlay_log_enabled = False
    alignment_overlay_log_num_frames = 30
    alignment_overlay_log_env_id = 0
    alignment_overlay_log_dir: str | None = None

    # Real-fish point cloud sequence used for imitation.
    real_pointcloud_dir = str(DEFAULT_REAL_POINTCLOUD_DIR)
    real_pointcloud_glob = "*.npy"
    real_pointcloud_frame_stride = 1
    real_pointcloud_loop = True
    real_pointcloud_random_start = False
    real_pointcloud_start_frame = 0

    # Chamfer reward settings.
    il_warmup_seconds = 3.0
    il_post_warmup_episode_seconds = 9.0
    chamfer_reward_weight = 1.0
    chamfer_point_count = 256
    centerline_num_bins = 21
    centerline_min_bin_points = 8

    @configclass
    class ObservationScalesCfg:
        joint_pos = 1.0
        joint_vel = 0.05
        root_lin_vel = 0.2
        root_ang_vel = 0.2

    obs_scales: ObservationScalesCfg = ObservationScalesCfg()

    target_root_height = 0.5
    termination_height = -1.0
    joint_limit_margin = 0.2
    # Post-warmup blow-up guards: these are intentionally loose and are only used
    # to catch clearly broken trajectories before PhysX turns the full state NaN.
    root_position_limit = 10.0
    joint_position_limit = 1.0
    root_linear_velocity_limit = 25.0
    root_angular_velocity_limit = 75.0
    joint_velocity_limit = 120.0

    initial_root_pos_range = (-0.0, 0.0)
    initial_root_rot_range = (-math.pi, math.pi)
    initial_joint_pos_range = (-0.0, 0.0)
    initial_joint_vel_range = (-0.0, 0.0)
    target_offset_xy = (2.0, 2.0)
    target_height = 0.05
