# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the salmon "imitate a real fish swimming, in ANALYTIC water" RL task.

This task REUSES the proven-stable analytic-water + FEM physics of the swim task
(`SalmonSwimEnvCfg`: zero-gravity MuJoCo ellipsoid hydro, per-slice bone ellipsoids,
PD-position D6 control, the FEM-CFL-stable dt = 1/960 / decimation 32 config, the
FEM material create+bind cure) and SWAPS the reward: instead of "swim toward a
target point", the reward is the **negative Chamfer distance between the simulated
FEM-fish point cloud and a real-fish video point cloud** (one of 451 frames in
`pointclouds/fg-*.npy`, 10000x3 each, looped). The fish is thus rewarded for
reproducing the real fish's body shape / undulation gait, and -- because it runs in
analytic water -- that imitated undulation actually propels it.

Everything physics-related is inherited unchanged from `SalmonSwimEnvCfg`. Only the
reward / observation / termination and the IL knobs below are new.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .salmon_swim_cfg import SalmonSwimEnvCfg

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)



# 451 real-fish frames live in <repo>/pointclouds (git-ignored, already present locally).
DEFAULT_REAL_POINTCLOUD_DIR = Path(__file__).resolve().parents[6] / "pointclouds"


@configclass
class SalmonILWaterEnvCfg(SalmonSwimEnvCfg):
    # ---- inherit the entire stable analytic-water + FEM physics from the swim cfg ----
    # (SIM_DT = 1/960, decimation = 32 -> control 30 Hz, episode = 1500 control steps,
    #  zero gravity, per-slice hydro, FEM material cure, PD-position D6 control.)

    # The FEM body MUST be active: the imitation reward reads the deformable simulation
    # mesh as the sim-fish point cloud. reset_deformable=True ALSO makes the base env
    # build the `_soft_view` DeformablePrim that we reuse to read those nodal positions.
    with_deformable = True
    reset_deformable = True
    # no target point in imitation -> no red marker
    draw_target_marker = False

    # Chamfer reward is an O(num_envs) python loop (per-env cdist + first-frame SVD
    # alignment), so keep the env count modest like the original salmon_IL task.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=8.0,
        replicate_physics=False,   # salmon USD has attachments PhysX cannot replicate
    )

    # No top-down camera video / trajectory PNGs for IL (avoid the record+FEM Replicator
    # stall and the per-episode file clutter; the chamfer/reward logs are the signal).
    record_video = False
    record_trajectory = False

    # ---- real-fish point cloud sequence (the imitation target) ----
    real_pointcloud_dir = str(DEFAULT_REAL_POINTCLOUD_DIR)
    real_pointcloud_glob = "*.npy"
    real_pointcloud_frame_stride = 1
    real_pointcloud_loop = True            # wrap the 451-frame clip modulo its length
    real_pointcloud_random_start = False   # every episode starts at frame `start_frame`
    real_pointcloud_start_frame = 0

    # ---- Chamfer imitation reward ----
    # Let the FEM body settle before scoring (the first frames after a reset are a
    # transient as the soft mesh relaxes onto the bones). warmup measured in SECONDS,
    # converted to control steps with step_dt = decimation * SIM_DT = 1/30 s.
    il_warmup_seconds = 1.0
    # BASELINE-IMPROVEMENT reward. The chamfer at the FIRST post-warmup step of each episode is
    # frozen as that episode's baseline (per env); the reward then pays ONLY for the improvement
    # BELOW that baseline. So the episode's own starting pose earns 0 and the policy is rewarded
    # only for tracking the real fish's gait BETTER than where it began -- this removes the "a
    # straight rest pose already scores chamfer ~0.02 -> ~full reward for doing nothing" plateau
    # that kept the fish from undulating.
    #   relative (default): reward = weight * clamp((baseline - chamfer)/baseline, 0, 1)  in [0,weight]
    #   absolute:           reward = weight * clamp( baseline - chamfer,          0)
    chamfer_reward_weight = 1.0
    chamfer_reward_relative = True
    # baseline = mean of the first N post-warmup chamfers (de-noises the single-frame baseline).
    # During those N frames reward = 0; then the baseline is frozen for the rest of the episode.
    baseline_num_frames = 5
    chamfer_point_count = 256             # both clouds are uniformly down-sampled to this

    # ---- DEMO-imitation mode (world-frame matching of a recorded RL rollout) ----
    # If demo_path is set, the reward switches to: WORLD-FRAME chamfer between the IL fish's FEM
    # point cloud and a recorded RL-demo fish's point cloud at the SAME timestep -- NO alignment.
    # The clouds must overlap in world coordinates, so the IL fish must match the demo's POSITION
    # AND pose. Same USD fish on both sides -> chamfer ~0 is achievable, so this is a clean
    # feasibility test of the whole IL/chamfer pipeline (ideally the IL policy recovers the RL one).
    # Demo npz from: scripts/rl_games/play.py --record_demo  (arrays: pc (T,N,3) world, root (T,13)).
    demo_path = ""
    # near-field shape term. "rational" = 1/(1+chamfer/temp): bounded [0,1], =1 at chamfer 0, but a
    # heavy (polynomial) tail so the gradient NEVER collapses to ~0 -> the policy gets a usable
    # "get closer" signal at any distance (unlike exp, which cliffs in the far field). "exp" =
    # exp(-chamfer/temp) (sharper near 0, dead far away).
    demo_reward_kind = "rational"        # "rational" (1/(1+c/temp)) | "exp" (exp(-c/temp))
    demo_reward_temp = 0.1                # length scale for the near-field term  (chamfer in metres)
    # FAR-FIELD penalty: exp(-chamfer/temp) saturates to ~0 once the clouds stop overlapping
    # (no gradient when the IL fish falls far behind the demo). Add a LINEAR penalty on the world
    # distance between the two clouds' centroids -> an always-on gradient that pulls them together
    # even with zero overlap. reward = exp(-chamfer/temp) - demo_far_weight * ||centroid_sim - centroid_demo||.
    demo_far_weight = 0.3
    # Reference State Initialization (RSI): on reset, set the fish to a RANDOM demo frame's FULL
    # state (root pose+vel, joint pos+vel, FEM nodal pos+vel) instead of always frame 0, so the
    # policy learns every part of the trajectory locally (the DeepMimic trick). Needs a demo
    # recorded with full state (joint_pos/joint_vel/nodal_vel); auto-disabled if absent.
    demo_rsi = True
    # Early Termination (ET): end the episode if the fish drifts further than this from the demo's
    # current position (centroid distance, metres). Keeps learning on the demo manifold. 0 disables.
    demo_et_centroid = 0.4
    # ---- velocity + orientation(angular-velocity) reward terms (point-cloud-derived) ----
    # Per-step ACTION-coupled signals the chamfer lacks. Linear: centroid velocity of the cloud.
    # Angular: rotation rate of the head->tail body axis. The head/tail end of the cloud is labelled
    # ONCE using the skeleton bone order (so the axis sign is anatomically consistent; real-world
    # clouds are assumed pre-labelled head/tail). reward += w_v*exp(-||dv||^2/tv) + w_a*exp(-||dw||^2/ta).
    demo_vel_weight = 1.0
    demo_vel_temp = 0.1          # (m/s)^2 scale
    demo_avel_weight = 1.0
    demo_avel_temp = 0.5         # (unit-axis/s)^2 scale
    # Egocentric MOVING-REFERENCE target in the OBSERVATION (+6 dims): the body-frame vector from
    # the agent to the demo's CURRENT position, plus the demo's heading expressed in the agent's
    # body frame. This makes the position/orientation the (existing) reward depends on OBSERVABLE,
    # so the policy can close the loop and track -- the de-aliasing fix that does NOT touch the
    # reward. It is a DIFFERENCE of positions, so translation-invariant (the absolute world frame
    # cancels). Reference-conditioned: needs the reference trajectory available at deployment.
    demo_target_obs = True
    centerline_num_bins = 21              # spine extraction (alignment) bin count
    centerline_min_bin_points = 8

    # per-step chamfer/reward print (noisy; off by default)
    debug_logs = False

    # ---- monitoring / outcome dir ----
    # Everything for a run lands in <outcome_dir>/run_<timestamp>/:
    #   trajectory_env0/   -> env0 path: a live traj_live.{png,obj} refreshed every
    #                         `traj_save_every` control steps, plus a kept traj_ep{N}.{png,obj}
    #                         snapshot at each env0 episode end (previous saves are KEPT).
    #   overlay_grid_ep{E}_{a}-{b}.png -> a grid of env0's real(blue) vs sim(orange) ALIGNED
    #                         point clouds (the per-step Chamfer comparison the policy sees), one
    #                         image per `overlay_window` post-warmup frames. Only the LATEST epoch's
    #                         overlay images are kept on disk (older epochs are purged).
    monitor_enabled = True
    outcome_dir = _P("${FISH_ROOT}/outcome")
    traj_save_every = 5            # control steps between env0 trajectory live writes (obj+png)
    overlay_window = 30            # env0 RECORDED frames per overlay-grid image (0-30, 30-60, ...)
    overlay_stride = 4             # record one overlay frame every N control steps (so consecutive
                                   # grid cells are visibly different, not 1 step / ~1mm apart)
    overlay_epoch_frames = 256     # env0 recorded frames per "epoch"; only the latest epoch is kept
    overlay_projection = "xy"      # body-plane projection for the overlay grid: "xy" | "xz" | "yz"
