# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform

from .salmon_IL_cfg import SalmonILEnvCfg, WATER_ENV_STAGE_PATH


class SalmonILEnv(DirectRLEnv):
    cfg: SalmonILEnvCfg

    def __init__(self, cfg: SalmonILEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self._soft_joint_limits = self.robot.data.soft_joint_pos_limits
        self._default_joint_pos = self.robot.data.default_joint_pos
        self._torques = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self._target_offset_xy = torch.tensor(self.cfg.target_offset_xy, dtype=torch.float32, device=self.device)
        self._target_height = torch.tensor(self.cfg.target_height, dtype=torch.float32, device=self.device)
        self.target_positions_w = self._compute_target_positions_world()
        self._action_log_counter = 0
        self._debug_log_limit = 5
        self._pending_joint_log = False
        self._pending_joint_log_step = 0

        self._episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self._best_chamfer = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._episode_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_reward_queue = deque(maxlen=100)
        self._cumulative_reward_queue = deque(maxlen=100)

        self._nan_reported = False
        self._action_nan_reported = False
        self._action_range_reported = False
        self._torque_nan_reported = False
        self._reward_nan_reported = False
        self._joint_violation_reported = False
        self._pointcloud_nan_reported = False
        self._alignment_nan_reported = False
        self._state_nan_reported = False
        self._reward_pointcloud_invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._pointcloud_capture_enabled = False
        self._pointcloud_capture_interval_steps = 0
        self._next_pointcloud_capture_step = 0
        self._pointcloud_capture_env_id = 0
        self._pointcloud_capture_count = 0
        self._pointcloud_capture_dir: Path | None = None
        self._pointcloud_mesh_prims = []
        self._pointcloud_deformable_prims = []
        self._pointcloud_deformable_views = []
        self._pointcloud_deformable_vertex_counts = []
        self._pointcloud_deformable_surface_node_indices = []
        self._pointcloud_deformable_surface_weights = []

        self._chamfer_reward_weight = float(self.cfg.chamfer_reward_weight)
        self._il_reward_point_count = max(1, int(self.cfg.chamfer_point_count))
        self._il_warmup_steps = max(0, int(round(float(self.cfg.il_warmup_seconds) / float(self.step_dt))))
        self._il_post_warmup_episode_steps = max(
            0, int(round(float(self.cfg.il_post_warmup_episode_seconds) / float(self.step_dt)))
        )
        self._il_env_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._real_pointcloud_sequence: torch.Tensor | None = None
        self._real_pointcloud_frame_count = 0
        self._il_real_frame_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._il_episode_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._chamfer_baseline = torch.zeros(self.num_envs, device=self.device)
        self._chamfer_baseline_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._current_chamfer = torch.zeros(self.num_envs, device=self.device)
        identity_rotation = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._alignment_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._real_alignment_midpoint = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._real_alignment_rotation = identity_rotation.clone()
        self._real_alignment_scale = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self._sim_alignment_midpoint = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._sim_alignment_rotation = identity_rotation.clone()
        self._sim_alignment_scale = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self._sim_alignment_flip_major = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._sim_reward_deformable_view = None
        self._sim_reward_vertex_count = 0
        self._sim_reward_surface_node_indices: torch.Tensor | None = None
        self._sim_reward_surface_weights: torch.Tensor | None = None
        self._sim_reward_sample_indices: torch.Tensor | None = None
        self._alignment_overlay_log_enabled = False
        self._alignment_overlay_log_num_frames = 0
        self._alignment_overlay_log_env_id = 0
        self._alignment_overlay_log_dir: Path | None = None
        self._alignment_overlay_log_saved = False
        self._alignment_overlay_records: list[dict] = []

        self._initialize_pointcloud_capture()
        self._initialize_imitation_reward_pipeline()
        self._initialize_alignment_overlay_logging()

    def _setup_scene(self):
        UsdGeom.SetStageMetersPerUnit(self.scene.stage, 1)
        self.robot = Articulation(self.cfg.robot_cfg)
        self._offset_source_asset_spawn_height()

        env0_scene_path = f"{self.scene.env_ns}/env_0/environment_scene"
        environment_cfg = sim_utils.UsdFileCfg(usd_path=str(WATER_ENV_STAGE_PATH))
        environment_cfg.func(prim_path=env0_scene_path, cfg=environment_cfg)
        self._configure_water_environment(env0_scene_path)
        self._disable_internal_rigid_collisions()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu" or not self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _configure_water_environment(self, env_scene_path: str) -> None:
        particles_enabled = bool(getattr(self.cfg, "water_particles_enabled", True))
        max_samples = int(getattr(self.cfg, "water_particle_max_samples", 0))
        sampling_distance = float(getattr(self.cfg, "water_particle_sampling_distance", 0.0))
        physics_scene_path = self._find_physics_scene_path()

        updated_any = False
        for prim in self.scene.stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(env_scene_path):
                continue
            if prim.GetTypeName() == "PhysxParticleSystem":
                particle_system_enabled_attr = prim.GetAttribute("particleSystemEnabled")
                if particle_system_enabled_attr and particle_system_enabled_attr.IsValid():
                    current_value = particle_system_enabled_attr.Get()
                    if current_value != particles_enabled:
                        particle_system_enabled_attr.Set(particles_enabled)
                        if self.cfg.debug_logs:
                            print(
                                f"[SalmonILEnv] set particle system enabled for {prim_path} "
                                f"from {current_value} to {particles_enabled}"
                            )
                    updated_any = True

                isosurface_attr = prim.GetAttribute("physxParticleIsosurface:isosurfaceEnabled")
                if isosurface_attr and isosurface_attr.IsValid():
                    desired_value = particles_enabled
                    current_value = isosurface_attr.Get()
                    if current_value != desired_value:
                        isosurface_attr.Set(desired_value)
                        if self.cfg.debug_logs:
                            print(
                                f"[SalmonILEnv] set particle isosurface for {prim_path} "
                                f"from {current_value} to {desired_value}"
                            )
                    updated_any = True

                if physics_scene_path is not None:
                    simulation_owner_rel = prim.GetRelationship("simulationOwner")
                    if simulation_owner_rel and simulation_owner_rel.IsValid():
                        desired_targets = [physics_scene_path]
                        current_targets = simulation_owner_rel.GetTargets()
                        if current_targets != desired_targets:
                            simulation_owner_rel.SetTargets(desired_targets)
                            if self.cfg.debug_logs:
                                print(
                                    f"[SalmonILEnv] bound particle system {prim_path} "
                                    f"to physics scene {physics_scene_path}"
                                )
                            updated_any = True

            if particles_enabled and prim.GetTypeName() == "Points":
                particle_system_rel = prim.GetRelationship("physxParticle:particleSystem")
                if particle_system_rel and particle_system_rel.IsValid():
                    points_attr = prim.GetAttribute("points")
                    points = points_attr.Get() if points_attr and points_attr.IsValid() else None
                    point_count = len(points) if points is not None else 0
                    if max_samples > 0 and point_count > max_samples:
                        sample_indices = np.linspace(0, point_count - 1, num=max_samples, dtype=np.int64)

                        def _downsample_attr(attr_name: str) -> None:
                            attr = prim.GetAttribute(attr_name)
                            if not attr or not attr.IsValid():
                                return
                            values = attr.Get()
                            if values is None or len(values) != point_count:
                                return
                            attr.Set(type(values)([values[int(i)] for i in sample_indices]))

                        _downsample_attr("points")
                        _downsample_attr("velocities")
                        _downsample_attr("widths")
                        if self.cfg.debug_logs:
                            print(
                                f"[SalmonILEnv] downsampled authored particle set for {prim_path} "
                                f"from {point_count} to {max_samples}"
                            )
                        updated_any = True

            if prim.GetName() == "Water":
                imageable = UsdGeom.Imageable(prim)
                visibility_attr = imageable.GetVisibilityAttr()
                if visibility_attr.IsValid():
                    current_visibility = visibility_attr.Get()
                    desired_visibility = UsdGeom.Tokens.invisible if particles_enabled else UsdGeom.Tokens.inherited
                    if current_visibility != desired_visibility:
                        visibility_attr.Set(desired_visibility)
                        if self.cfg.debug_logs:
                            print(
                                f"[SalmonILEnv] set water visibility for {prim_path} "
                                f"from {current_visibility} to {desired_visibility}"
                            )
                    updated_any = True

                volume_attr = prim.GetAttribute("physxParticleSampling:volume")
                if volume_attr and volume_attr.IsValid():
                    current_value = volume_attr.Get()
                    desired_value = False
                    if current_value != desired_value:
                        volume_attr.Set(desired_value)
                        if self.cfg.debug_logs:
                            print(
                                f"[SalmonILEnv] set particle volume sampling for {prim_path} "
                                f"from {current_value} to {desired_value}"
                            )
                    updated_any = True

            if max_samples > 0:
                max_samples_attr = prim.GetAttribute("physxParticleSampling:maxSamples")
                if max_samples_attr and max_samples_attr.IsValid():
                    current_value = max_samples_attr.Get()
                    if current_value != max_samples:
                        max_samples_attr.Set(max_samples)
                        if self.cfg.debug_logs:
                            print(
                                f"[SalmonILEnv] capped particle sampling for {prim_path} "
                                f"from {current_value} to {max_samples}"
                            )
                    updated_any = True

            if sampling_distance > 0.0:
                sampling_distance_attr = prim.GetAttribute("physxParticleSampling:samplingDistance")
                if sampling_distance_attr and sampling_distance_attr.IsValid():
                    current_value = sampling_distance_attr.Get()
                    if current_value != sampling_distance:
                        sampling_distance_attr.Set(sampling_distance)
                        if self.cfg.debug_logs:
                            print(
                                f"[SalmonILEnv] set particle sampling distance for {prim_path} "
                                f"from {current_value} to {sampling_distance}"
                            )
                    updated_any = True

        if not updated_any and self.cfg.debug_logs:
            print(f"[SalmonILEnv] no particle-sampled prims found under {env_scene_path}")

    def _find_physics_scene_path(self) -> Sdf.Path | None:
        for prim in self.scene.stage.Traverse():
            if prim.GetTypeName() == "PhysicsScene":
                return prim.GetPath()
        if self.cfg.debug_logs:
            print("[SalmonILEnv] no PhysicsScene found while configuring water environment")
        return None

    def _disable_internal_rigid_collisions(self) -> None:
        if not bool(getattr(self.cfg, "disable_internal_rigid_collisions", False)):
            return

        asset_prefix = f"{self.scene.env_ns}/env_0/skeleton"
        disabled_paths: list[str] = []
        for prim in self.scene.stage.Traverse():
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(asset_prefix):
                continue
            if "/deformable_salmon/" in prim_path:
                continue
            collision_api = UsdPhysics.CollisionAPI(prim)
            if not collision_api:
                continue
            collision_enabled_attr = collision_api.GetCollisionEnabledAttr()
            if not collision_enabled_attr.IsValid():
                continue
            if collision_enabled_attr.Get() is False:
                continue
            collision_enabled_attr.Set(False)
            disabled_paths.append(prim_path)

        if self.cfg.debug_logs:
            if disabled_paths:
                preview = ", ".join(disabled_paths[:3])
                suffix = "..." if len(disabled_paths) > 3 else ""
                print(
                    f"[SalmonILEnv] disabled {len(disabled_paths)} internal rigid collisions "
                    f"under {asset_prefix}: {preview}{suffix}"
                )
            else:
                print(f"[SalmonILEnv] no internal rigid collisions found under {asset_prefix}")


    def _offset_source_asset_spawn_height(self) -> None:
        height_offset = float(self.cfg.asset_spawn_height_offset)
        if height_offset == 0.0:
            return

        source_asset_path = f"{self.scene.env_ns}/env_0/skeleton"
        source_prim = self.scene.stage.GetPrimAtPath(source_asset_path)
        if not source_prim:
            if self.cfg.debug_logs:
                print(f"[SalmonILEnv] source asset prim not found at {source_asset_path}")
            return

        xformable = UsdGeom.Xformable(source_prim)
        translate_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and op.GetOpName() == "xformOp:translate":
                translate_op = op
                break
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()

        current = translate_op.Get()
        if current is None:
            current = Gf.Vec3d(0.0, 0.0, 0.0)
        translate_op.Set(Gf.Vec3d(float(current[0]), float(current[1]), float(current[2]) + height_offset))
        if self.cfg.debug_logs:
            print(f"[SalmonILEnv] lifted source asset {source_asset_path} by {height_offset}m")

    def _initialize_pointcloud_capture(self) -> None:
        self._pointcloud_capture_enabled = bool(self.cfg.pointcloud_capture_enabled)
        if not self._pointcloud_capture_enabled:
            return

        interval_s = float(self.cfg.pointcloud_capture_interval_s)
        if interval_s <= 0.0:
            self._pointcloud_capture_enabled = False
            return

        self._pointcloud_capture_interval_steps = max(1, int(round(interval_s / float(self.step_dt))))
        self._next_pointcloud_capture_step = self._pointcloud_capture_interval_steps
        self._pointcloud_capture_env_id = int(self.cfg.pointcloud_capture_env_id)
        self._pointcloud_capture_env_id = max(0, min(self._pointcloud_capture_env_id, self.num_envs - 1))

        output_dir = self.cfg.pointcloud_capture_dir
        if output_dir:
            self._pointcloud_capture_dir = Path(output_dir)
        else:
            self._pointcloud_capture_dir = Path(__file__).resolve().parents[6] / "logs" / "fish_pointclouds"
        self._pointcloud_capture_dir.mkdir(parents=True, exist_ok=True)

        self._pointcloud_deformable_prims = self._collect_pointcloud_deformable_prims(self._pointcloud_capture_env_id)
        self._pointcloud_deformable_views = []
        self._pointcloud_deformable_vertex_counts = []
        self._pointcloud_deformable_surface_node_indices = []
        self._pointcloud_deformable_surface_weights = []
        if self._pointcloud_deformable_prims:
            self._initialize_deformable_pointcloud_capture()

        self._pointcloud_mesh_prims = self._collect_pointcloud_mesh_prims(self._pointcloud_capture_env_id)
        if not self._pointcloud_deformable_views and not self._pointcloud_mesh_prims:
            self._pointcloud_capture_enabled = False
            return

        capture_source = "usd_mesh_stage"
        if self._pointcloud_deformable_views:
            capture_source = "deformable_simulation_mesh"
            if any(node_ids is not None for node_ids in self._pointcloud_deformable_surface_node_indices):
                capture_source = "deformable_render_surface_embedded"
        metadata = {
            "env_id": self._pointcloud_capture_env_id,
            "interval_s": interval_s,
            "interval_steps": self._pointcloud_capture_interval_steps,
            "step_dt": float(self.step_dt),
            "mesh_paths": [str(prim.GetPath()) for prim in self._pointcloud_mesh_prims],
            "deformable_paths": [str(prim.GetPath()) for prim in self._pointcloud_deformable_prims],
            "capture_source": capture_source,
        }
        metadata_path = self._pointcloud_capture_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _collect_pointcloud_deformable_prims(self, env_id: int) -> list:
        asset_prefix = f"{self.scene.env_ns}/env_{env_id}/skeleton"
        deformable_prims = []
        for prim in self.scene.stage.Traverse():
            if prim.GetTypeName() != "Mesh":
                continue
            prim_path = str(prim.GetPath())
            if not prim_path.startswith(asset_prefix):
                continue
            simulation_points_attr = prim.GetAttribute("physxDeformable:simulationPoints")
            if simulation_points_attr and simulation_points_attr.IsValid():
                deformable_prims.append(prim)
        deformable_prims.sort(key=lambda prim: str(prim.GetPath()))
        return deformable_prims

    def _initialize_deformable_pointcloud_capture(self) -> None:
        try:
            from isaacsim.core.prims import DeformablePrim
        except Exception as exc:
            print(f"[SalmonILEnv] deformable point cloud capture unavailable (import failed: {exc})")
            return

        active_prims = []
        for view_id, prim in enumerate(self._pointcloud_deformable_prims):
            prim_path = str(prim.GetPath())
            try:
                deformable_view = DeformablePrim(
                    prim_paths_expr=prim_path,
                    name=f"salmon_il_pointcloud_view_{view_id}",
                    reset_xform_properties=False,
                )
                deformable_view.initialize()
                sim_points = prim.GetAttribute("physxDeformable:simulationPoints").Get()
                sim_vertex_count = len(sim_points) if sim_points is not None else 0
                if deformable_view.count < 1 or sim_vertex_count < 1:
                    continue
                surface_node_indices, surface_weights = self._build_deformable_surface_embedding(prim)
                active_prims.append(prim)
                self._pointcloud_deformable_views.append(deformable_view)
                self._pointcloud_deformable_vertex_counts.append(sim_vertex_count)
                self._pointcloud_deformable_surface_node_indices.append(surface_node_indices)
                self._pointcloud_deformable_surface_weights.append(surface_weights)
            except Exception as exc:
                print(f"[SalmonILEnv] deformable point cloud capture unavailable for {prim_path} ({exc})")
        self._pointcloud_deformable_prims = active_prims

    def _build_deformable_surface_embedding(self, prim) -> tuple[np.ndarray | None, np.ndarray | None]:
        simulation_rest_points_attr = prim.GetAttribute("physxDeformable:simulationRestPoints")
        simulation_indices_attr = prim.GetAttribute("physxDeformable:simulationIndices")
        surface_rest_points_attr = prim.GetAttribute("physxDeformable:restPoints")
        if not simulation_rest_points_attr or not simulation_rest_points_attr.IsValid():
            return None, None
        if not simulation_indices_attr or not simulation_indices_attr.IsValid():
            return None, None
        if not surface_rest_points_attr or not surface_rest_points_attr.IsValid():
            return None, None

        simulation_rest_points = simulation_rest_points_attr.Get()
        simulation_indices = simulation_indices_attr.Get()
        surface_rest_points = surface_rest_points_attr.Get()
        if simulation_rest_points is None or simulation_indices is None or surface_rest_points is None:
            return None, None

        sim_rest = np.asarray(simulation_rest_points, dtype=np.float64)
        tet_indices = np.asarray(simulation_indices, dtype=np.int32).reshape(-1, 4)
        surface_rest = np.asarray(surface_rest_points, dtype=np.float64)
        if sim_rest.size == 0 or tet_indices.size == 0 or surface_rest.size == 0:
            return None, None

        node_indices, weights, _, _ = self._compute_tetrahedral_surface_embedding(surface_rest, sim_rest, tet_indices)
        return node_indices, weights

    @staticmethod
    def _compute_tetrahedral_surface_embedding(
        surface_rest_points: np.ndarray,
        simulation_rest_points: np.ndarray,
        tet_indices: np.ndarray,
        tol: float = 1.0e-6,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        tet_points = simulation_rest_points[tet_indices]
        tet_origin = tet_points[:, 3, :]
        tet_mats = np.transpose(tet_points[:, :3, :] - tet_origin[:, None, :], (0, 2, 1))
        tet_inv_mats = np.linalg.pinv(tet_mats)

        node_indices = np.empty((surface_rest_points.shape[0], 4), dtype=np.int32)
        weights = np.empty((surface_rest_points.shape[0], 4), dtype=np.float32)
        inside_count = 0
        max_error = 0.0

        for point_id, point in enumerate(surface_rest_points):
            rhs = point[None, :] - tet_origin
            uvw = np.einsum("tij,tj->ti", tet_inv_mats, rhs, optimize=True)
            bary = np.concatenate((uvw, 1.0 - uvw.sum(axis=1, keepdims=True)), axis=1)
            inside_mask = np.all(bary >= -tol, axis=1) & np.all(bary <= 1.0 + tol, axis=1)
            scores = bary.min(axis=1)
            if inside_mask.any():
                inside_ids = np.flatnonzero(inside_mask)
                tet_id = int(inside_ids[np.argmax(scores[inside_mask])])
                inside_count += 1
            else:
                tet_id = int(np.argmax(scores))

            bary_weights = bary[tet_id]
            bary_weights = bary_weights / bary_weights.sum()
            node_indices[point_id] = tet_indices[tet_id]
            weights[point_id] = bary_weights.astype(np.float32)

            reconstructed = (tet_points[tet_id] * bary_weights[:, None]).sum(axis=0)
            max_error = max(max_error, float(np.linalg.norm(reconstructed - point)))

        return node_indices, weights, inside_count, max_error

    def _collect_pointcloud_mesh_prims(self, env_id: int) -> list:
        asset_prefix = f"{self.scene.env_ns}/env_{env_id}/skeleton"
        mesh_prims = []
        for prim in self.scene.stage.Traverse():
            if prim.GetTypeName() != "Mesh":
                continue
            prim_path = str(prim.GetPath())
            if prim_path.startswith(asset_prefix):
                mesh_prims.append(prim)
        mesh_prims.sort(key=lambda prim: str(prim.GetPath()))
        return mesh_prims

    def _maybe_capture_pointcloud(self) -> None:
        if not self._pointcloud_capture_enabled:
            return
        if self.common_step_counter < self._next_pointcloud_capture_step:
            return

        world_point_sets = []
        mesh_paths = []
        mesh_counts = []
        if self._pointcloud_deformable_views:
            for deformable_view, prim, sim_vertex_count, surface_node_indices, surface_weights in zip(
                self._pointcloud_deformable_views,
                self._pointcloud_deformable_prims,
                self._pointcloud_deformable_vertex_counts,
                self._pointcloud_deformable_surface_node_indices,
                self._pointcloud_deformable_surface_weights,
                strict=False,
            ):
                try:
                    positions = deformable_view.get_simulation_mesh_nodal_positions()
                    if isinstance(positions, torch.Tensor):
                        live_positions = positions[0, :sim_vertex_count].detach().to("cpu").numpy().astype(np.float32)
                    else:
                        live_positions = np.asarray(positions[0, :sim_vertex_count], dtype=np.float32)
                    if surface_node_indices is not None and surface_weights is not None:
                        embedded_live_positions = live_positions[surface_node_indices]
                        point_array = np.einsum(
                            "vi,vij->vj", surface_weights, embedded_live_positions, optimize=True
                        ).astype(np.float32, copy=False)
                    else:
                        point_array = live_positions
                except Exception as exc:
                    print(f"[SalmonILEnv] deformable point cloud read failed for {prim.GetPath()} ({exc})")
                    continue
                if point_array.size == 0:
                    continue
                world_point_sets.append(point_array)
                mesh_paths.append(str(prim.GetPath()))
                mesh_counts.append(point_array.shape[0])
        else:
            xform_cache = UsdGeom.XformCache()
            for prim in self._pointcloud_mesh_prims:
                if not prim.IsValid():
                    continue
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                if not points:
                    continue
                transform = xform_cache.GetLocalToWorldTransform(prim)
                point_array = np.empty((len(points), 3), dtype=np.float32)
                for point_id, point in enumerate(points):
                    world_point = transform.Transform(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
                    point_array[point_id] = (float(world_point[0]), float(world_point[1]), float(world_point[2]))
                world_point_sets.append(point_array)
                mesh_paths.append(str(prim.GetPath()))
                mesh_counts.append(point_array.shape[0])

        if not world_point_sets:
            return

        sim_time_s = float(self.common_step_counter * self.step_dt)
        points_world = np.concatenate(world_point_sets, axis=0)
        capture_path = self._pointcloud_capture_dir / (
            f"fish_env{self._pointcloud_capture_env_id:03d}_"
            f"step{self.common_step_counter:07d}_"
            f"t{sim_time_s:08.3f}s.npz"
        )
        np.savez_compressed(
            capture_path,
            points=points_world,
            mesh_paths=np.asarray(mesh_paths, dtype=str),
            mesh_point_counts=np.asarray(mesh_counts, dtype=np.int32),
            env_id=np.asarray([self._pointcloud_capture_env_id], dtype=np.int32),
            step=np.asarray([self.common_step_counter], dtype=np.int64),
            sim_time_s=np.asarray([sim_time_s], dtype=np.float64),
        )
        self._pointcloud_capture_count += 1
        self._next_pointcloud_capture_step += self._pointcloud_capture_interval_steps

    def _initialize_imitation_reward_pipeline(self) -> None:
        self._initialize_simulation_reward_pointclouds()
        self._load_real_pointcloud_sequence()

    def _initialize_alignment_overlay_logging(self) -> None:
        self._alignment_overlay_log_enabled = bool(self.cfg.alignment_overlay_log_enabled)
        if not self._alignment_overlay_log_enabled:
            return

        self._alignment_overlay_log_num_frames = max(1, int(self.cfg.alignment_overlay_log_num_frames))
        self._alignment_overlay_log_env_id = int(self.cfg.alignment_overlay_log_env_id)
        self._alignment_overlay_log_env_id = max(0, min(self._alignment_overlay_log_env_id, self.num_envs - 1))
        if self.cfg.alignment_overlay_log_dir:
            self._alignment_overlay_log_dir = Path(self.cfg.alignment_overlay_log_dir)
        elif self.cfg.pointcloud_capture_dir:
            self._alignment_overlay_log_dir = Path(self.cfg.pointcloud_capture_dir) / "reward_alignment"
        else:
            self._alignment_overlay_log_dir = Path(__file__).resolve().parents[6] / "logs" / "salmon_il_reward_alignment"
        self._alignment_overlay_log_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_record_alignment_overlay_frame(
        self,
        env_id: int,
        real_frame_id: int,
        real_aligned: torch.Tensor,
        sim_aligned: torch.Tensor,
        chamfer: torch.Tensor,
    ) -> None:
        if not self._alignment_overlay_log_enabled or self._alignment_overlay_log_saved:
            return
        if env_id != self._alignment_overlay_log_env_id:
            return

        real_aligned = torch.nan_to_num(real_aligned, nan=0.0, posinf=0.0, neginf=0.0)
        sim_aligned = torch.nan_to_num(sim_aligned, nan=0.0, posinf=0.0, neginf=0.0)
        chamfer = torch.nan_to_num(chamfer, nan=1.0e6, posinf=1.0e6, neginf=0.0)

        self._alignment_overlay_records.append(
            {
                "frame": len(self._alignment_overlay_records),
                "real_frame": int(real_frame_id),
                "flip_major": bool(self._sim_alignment_flip_major[env_id].item()),
                "chamfer": float(chamfer.detach().to("cpu").item()),
                "real_aligned": real_aligned.detach().to("cpu").numpy(),
                "sim_aligned": sim_aligned.detach().to("cpu").numpy(),
            }
        )
        if len(self._alignment_overlay_records) >= self._alignment_overlay_log_num_frames:
            self._flush_alignment_overlay_logs(mark_saved=True)

    @staticmethod
    def _alignment_overlay_limits(records: list[dict], axes: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]]:
        all_real = np.concatenate([record["real_aligned"][:, axes] for record in records], axis=0)
        all_sim = np.concatenate([record["sim_aligned"][:, axes] for record in records], axis=0)
        all_points = np.concatenate((all_real, all_sim), axis=0)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        pad = np.maximum((maxs - mins) * 0.05, 1.0e-3)
        return (mins[0] - pad[0], maxs[0] + pad[0]), (mins[1] - pad[1], maxs[1] + pad[1])

    def _write_alignment_overlay_projection_grid(
        self,
        records: list[dict],
        axes: tuple[int, int],
        axis_labels: tuple[str, str],
        output_path: Path,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xlim, ylim = self._alignment_overlay_limits(records, axes)
        num_records = len(records)
        num_cols = min(6, max(1, int(np.ceil(np.sqrt(num_records)))))
        num_rows = int(np.ceil(num_records / num_cols))
        fig, grid = plt.subplots(num_rows, num_cols, figsize=(3.2 * num_cols, 2.8 * num_rows), constrained_layout=True)
        axes_array = np.atleast_1d(grid).reshape(-1)
        for ax, record in zip(axes_array, records, strict=False):
            real_points = record["real_aligned"][:, axes]
            sim_points = record["sim_aligned"][:, axes]
            ax.scatter(real_points[:, 0], real_points[:, 1], s=1.0, alpha=0.35, c="#1f77b4", label="real")
            ax.scatter(sim_points[:, 0], sim_points[:, 1], s=1.0, alpha=0.35, c="#ff7f0e", label="sim")
            ax.set_title(
                f"frame {record['frame']:02d} | real {record['real_frame']:03d}\n"
                f"Chamfer {record['chamfer']:.4f} | flip={record['flip_major']}",
                fontsize=8,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes_array[num_records:]:
            ax.axis("off")

        handles, labels = axes_array[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(
            f"Salmon IL reward-frame overlap ({axis_labels[0]}-{axis_labels[1]} projection)\n"
            "Real and simulated point clouds are shown in the fixed frame-0 reward coordinate used by the policy.",
            fontsize=14,
        )
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def _write_alignment_overlay_metrics(self, records: list[dict], output_path: Path) -> None:
        import csv

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["frame", "real_frame", "chamfer", "flip_major"])
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "frame": record["frame"],
                        "real_frame": record["real_frame"],
                        "chamfer": record["chamfer"],
                        "flip_major": record["flip_major"],
                    }
                )

    def _write_alignment_overlay_logs(self, mark_saved: bool) -> None:
        if not self._alignment_overlay_log_enabled or self._alignment_overlay_log_dir is None:
            return
        if not self._alignment_overlay_records:
            return

        records = list(self._alignment_overlay_records)
        frame_suffix = f"first{self._alignment_overlay_log_num_frames}"
        self._alignment_overlay_log_dir.mkdir(parents=True, exist_ok=True)
        self._write_alignment_overlay_projection_grid(
            records,
            (0, 1),
            ("x", "y"),
            self._alignment_overlay_log_dir / f"overlay_grid_xy_{frame_suffix}.png",
        )
        self._write_alignment_overlay_projection_grid(
            records,
            (0, 2),
            ("x", "z"),
            self._alignment_overlay_log_dir / f"overlay_grid_xz_{frame_suffix}.png",
        )
        self._write_alignment_overlay_projection_grid(
            records,
            (1, 2),
            ("y", "z"),
            self._alignment_overlay_log_dir / f"overlay_grid_yz_{frame_suffix}.png",
        )
        self._write_alignment_overlay_metrics(records, self._alignment_overlay_log_dir / "alignment_metrics.csv")
        if mark_saved:
            self._alignment_overlay_log_saved = True

    def _flush_alignment_overlay_logs(self, mark_saved: bool) -> None:
        try:
            self._write_alignment_overlay_logs(mark_saved=mark_saved)
        except Exception as exc:
            print(f"[SalmonILEnv] alignment overlay logging failed ({exc})")

    def _initialize_simulation_reward_pointclouds(self) -> None:
        try:
            from isaacsim.core.prims import DeformablePrim
        except Exception as exc:
            raise RuntimeError(f"Deformable reward view is unavailable: {exc}") from exc

        env0_deformable_prims = self._collect_pointcloud_deformable_prims(0)
        if not env0_deformable_prims:
            raise RuntimeError("No deformable fish mesh was found for the salmon imitation reward.")

        reward_prim = env0_deformable_prims[0]
        reward_prim_path = str(reward_prim.GetPath())
        reward_expr = reward_prim_path.replace("/env_0/", "/env_.*/")
        self._sim_reward_deformable_view = DeformablePrim(
            prim_paths_expr=reward_expr,
            name="salmon_il_reward_deformable_view",
            reset_xform_properties=False,
        )
        self._sim_reward_deformable_view.initialize()

        sim_points = reward_prim.GetAttribute("physxDeformable:simulationPoints").Get()
        self._sim_reward_vertex_count = len(sim_points) if sim_points is not None else 0
        if self._sim_reward_vertex_count < 1:
            raise RuntimeError(f"Deformable reward prim has no simulation points: {reward_prim_path}")

        surface_node_indices, surface_weights = self._build_deformable_surface_embedding(reward_prim)
        if surface_node_indices is None or surface_weights is None:
            sample_ids = self._uniform_sample_indices(self._sim_reward_vertex_count, self._il_reward_point_count)
            self._il_reward_point_count = len(sample_ids)
            self._sim_reward_sample_indices = torch.as_tensor(sample_ids, dtype=torch.long, device=self.device)
            return

        sample_ids = self._uniform_sample_indices(surface_node_indices.shape[0], self._il_reward_point_count)
        self._il_reward_point_count = len(sample_ids)
        self._sim_reward_surface_node_indices = torch.as_tensor(
            surface_node_indices[sample_ids], dtype=torch.long, device=self.device
        )
        self._sim_reward_surface_weights = torch.as_tensor(
            surface_weights[sample_ids], dtype=torch.float32, device=self.device
        )

    def _load_real_pointcloud_sequence(self) -> None:
        pointcloud_dir = Path(self.cfg.real_pointcloud_dir)
        if not pointcloud_dir.exists():
            raise RuntimeError(f"Real-fish point cloud directory does not exist: {pointcloud_dir}")

        frame_paths = sorted(pointcloud_dir.glob(self.cfg.real_pointcloud_glob))
        stride = max(1, int(self.cfg.real_pointcloud_frame_stride))
        frame_paths = frame_paths[::stride]
        if not frame_paths:
            raise RuntimeError(f"No real-fish point clouds were found under {pointcloud_dir}")

        frame_tensors = []
        for frame_path in frame_paths:
            frame_points = np.load(frame_path)
            if frame_points.ndim != 2 or frame_points.shape[1] != 3:
                raise RuntimeError(f"Unexpected point cloud shape in {frame_path}: {frame_points.shape}")
            if not np.isfinite(frame_points).all():
                frame_points = np.nan_to_num(frame_points, nan=0.0, posinf=0.0, neginf=0.0)
            sample_ids = self._uniform_sample_indices(frame_points.shape[0], self._il_reward_point_count)
            sampled_points = frame_points[sample_ids].astype(np.float32, copy=False)
            frame_tensors.append(torch.as_tensor(sampled_points, dtype=torch.float32, device=self.device))

        self._real_pointcloud_sequence = torch.stack(frame_tensors, dim=0)
        self._real_pointcloud_frame_count = self._real_pointcloud_sequence.shape[0]

    @staticmethod
    def _uniform_sample_indices(total_points: int, max_points: int) -> np.ndarray:
        if total_points <= 0:
            raise ValueError("Cannot sample from an empty point set.")
        if total_points <= max_points:
            return np.arange(total_points, dtype=np.int64)
        return np.linspace(0, total_points - 1, max_points, dtype=np.int64)

    def _get_real_frame_indices(self) -> torch.Tensor:
        frame_indices = self._il_real_frame_start + self._il_episode_frame
        if self.cfg.real_pointcloud_loop:
            return torch.remainder(frame_indices, self._real_pointcloud_frame_count)
        return torch.clamp(frame_indices, max=self._real_pointcloud_frame_count - 1)

    def _get_live_sim_pointclouds_for_reward(self) -> torch.Tensor:
        positions = self._sim_reward_deformable_view.get_simulation_mesh_nodal_positions()
        if not isinstance(positions, torch.Tensor):
            positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        else:
            positions = positions.to(self.device)
        positions = positions[: self.num_envs, : self._sim_reward_vertex_count]
        invalid_env_mask = (~torch.isfinite(positions)).any(dim=(1, 2))
        self._reward_pointcloud_invalid |= invalid_env_mask
        if not self._pointcloud_nan_reported and invalid_env_mask.any():
            env_ids = torch.nonzero(invalid_env_mask, as_tuple=False).flatten()
            env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
            print(f"[SalmonILEnv] non-finite deformable reward vertices detected env{env_id}; clamping to zero.")
            self._pointcloud_nan_reported = True
        positions = torch.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)

        if self._sim_reward_surface_node_indices is not None and self._sim_reward_surface_weights is not None:
            embedded_positions = positions[:, self._sim_reward_surface_node_indices, :]
            reward_points = torch.einsum("pf,epfj->epj", self._sim_reward_surface_weights, embedded_positions)
            return torch.nan_to_num(reward_points, nan=0.0, posinf=0.0, neginf=0.0)

        reward_points = positions[:, self._sim_reward_sample_indices, :]
        return torch.nan_to_num(reward_points, nan=0.0, posinf=0.0, neginf=0.0)

    def _extract_centerline(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        if points.shape[0] <= 1:
            return points

        centroid = points.mean(dim=0)
        centered = points - centroid
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            major_axis = vh[0]
        except RuntimeError:
            major_axis = torch.tensor([1.0, 0.0, 0.0], dtype=points.dtype, device=points.device)

        major_axis = major_axis / major_axis.norm().clamp(min=1.0e-6)
        projections = centered @ major_axis
        proj_min = projections.min()
        proj_max = projections.max()
        if float((proj_max - proj_min).abs().item()) < 1.0e-6:
            return torch.stack((points[0], points[-1]), dim=0)

        bin_edges = torch.linspace(proj_min, proj_max, steps=int(self.cfg.centerline_num_bins) + 1, device=points.device)
        centerline_points = []
        min_points = max(1, int(self.cfg.centerline_min_bin_points))
        for bin_id in range(len(bin_edges) - 1):
            lower = bin_edges[bin_id]
            upper = bin_edges[bin_id + 1]
            if bin_id == len(bin_edges) - 2:
                mask = (projections >= lower) & (projections <= upper)
            else:
                mask = (projections >= lower) & (projections < upper)
            if int(mask.sum().item()) < min_points:
                continue
            centerline_points.append(points[mask].mean(dim=0))

        if len(centerline_points) < 2:
            lo_idx = torch.argmin(projections)
            hi_idx = torch.argmax(projections)
            return torch.stack((points[lo_idx], points[hi_idx]), dim=0)

        centerline = torch.stack(centerline_points, dim=0)
        centerline_proj = (centerline - centroid) @ major_axis
        centerline = centerline[torch.argsort(centerline_proj)]
        return torch.nan_to_num(centerline, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _centerline_length(centerline: torch.Tensor) -> torch.Tensor:
        if centerline.shape[0] < 2:
            return torch.tensor(1.0, dtype=centerline.dtype, device=centerline.device)
        return torch.linalg.norm(centerline[1:] - centerline[:-1], dim=1).sum().clamp(min=1.0e-6)

    @staticmethod
    def _fallback_orthogonal_vector(vector: torch.Tensor) -> torch.Tensor:
        if abs(float(vector[0].item())) < 0.9:
            reference = torch.tensor([1.0, 0.0, 0.0], dtype=vector.dtype, device=vector.device)
        else:
            reference = torch.tensor([0.0, 1.0, 0.0], dtype=vector.dtype, device=vector.device)
        ortho = reference - torch.dot(reference, vector) * vector
        ortho_norm = ortho.norm()
        if float(ortho_norm.item()) < 1.0e-6:
            reference = torch.tensor([0.0, 0.0, 1.0], dtype=vector.dtype, device=vector.device)
            ortho = reference - torch.dot(reference, vector) * vector
            ortho_norm = ortho.norm()
        return ortho / ortho_norm.clamp(min=1.0e-6)

    def _compute_alignment_transform(
        self, points: torch.Tensor, flip_major: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        centerline = self._extract_centerline(points)
        midpoint = 0.5 * (centerline[0] + centerline[-1])

        centered = points - points.mean(dim=0)
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            vh = torch.eye(3, dtype=points.dtype, device=points.device)

        major_axis = centerline[-1] - centerline[0]
        if float(major_axis.norm().item()) < 1.0e-6:
            major_axis = vh[0]
        major_axis = major_axis / major_axis.norm().clamp(min=1.0e-6)
        if flip_major:
            major_axis = -major_axis

        secondary_axis = vh[1] if vh.shape[0] > 1 else self._fallback_orthogonal_vector(major_axis)
        secondary_axis = secondary_axis - torch.dot(secondary_axis, major_axis) * major_axis
        if float(secondary_axis.norm().item()) < 1.0e-6:
            secondary_axis = self._fallback_orthogonal_vector(major_axis)
        secondary_axis = secondary_axis / secondary_axis.norm().clamp(min=1.0e-6)

        third_axis = torch.cross(major_axis, secondary_axis, dim=0)
        if float(third_axis.norm().item()) < 1.0e-6:
            secondary_axis = self._fallback_orthogonal_vector(major_axis)
            third_axis = torch.cross(major_axis, secondary_axis, dim=0)
        third_axis = third_axis / third_axis.norm().clamp(min=1.0e-6)
        secondary_axis = torch.cross(third_axis, major_axis, dim=0)
        secondary_axis = secondary_axis / secondary_axis.norm().clamp(min=1.0e-6)

        rotation = torch.stack((major_axis, secondary_axis, third_axis), dim=1)
        body_length = self._centerline_length(centerline)
        raw_midpoint = midpoint
        raw_rotation = rotation
        raw_body_length = body_length
        midpoint = torch.nan_to_num(midpoint, nan=0.0, posinf=0.0, neginf=0.0)
        rotation = torch.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0)
        body_length = torch.nan_to_num(body_length, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1.0e-6)
        if not self._alignment_nan_reported and (
            (~torch.isfinite(raw_midpoint)).any()
            or (~torch.isfinite(raw_rotation)).any()
            or ~torch.isfinite(raw_body_length)
        ):
            print("[SalmonILEnv] non-finite alignment transform detected; clamping to safe values.")
            self._alignment_nan_reported = True
        return midpoint, rotation, body_length

    @staticmethod
    def _apply_alignment_transform(
        points: torch.Tensor, midpoint: torch.Tensor, rotation: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        aligned = (torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0) - midpoint) @ rotation
        aligned = aligned / scale.clamp(min=1.0e-6)
        return torch.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_reward_aligned_pointclouds(
        self, env_id: int, real_points: torch.Tensor, sim_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self._alignment_initialized[env_id].item()):
            raise RuntimeError(f"Reward alignment has not been initialized for env {env_id}.")
        real_aligned = self._apply_alignment_transform(
            real_points,
            self._real_alignment_midpoint[env_id],
            self._real_alignment_rotation[env_id],
            self._real_alignment_scale[env_id],
        )
        sim_aligned = self._apply_alignment_transform(
            sim_points,
            self._sim_alignment_midpoint[env_id],
            self._sim_alignment_rotation[env_id],
            self._sim_alignment_scale[env_id],
        )
        return real_aligned, sim_aligned

    def _initialize_episode_alignment(
        self, env_id: int, real_points: torch.Tensor, sim_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real_midpoint, real_rotation, real_scale = self._compute_alignment_transform(real_points, flip_major=False)
        real_aligned = self._apply_alignment_transform(real_points, real_midpoint, real_rotation, real_scale)

        sim_midpoint_direct, sim_rotation_direct, sim_scale_direct = self._compute_alignment_transform(
            sim_points, flip_major=False
        )
        sim_aligned_direct = self._apply_alignment_transform(
            sim_points, sim_midpoint_direct, sim_rotation_direct, sim_scale_direct
        )
        chamfer_direct = self._symmetric_chamfer_distance(real_aligned, sim_aligned_direct)

        sim_midpoint_flipped, sim_rotation_flipped, sim_scale_flipped = self._compute_alignment_transform(
            sim_points, flip_major=True
        )
        sim_aligned_flipped = self._apply_alignment_transform(
            sim_points, sim_midpoint_flipped, sim_rotation_flipped, sim_scale_flipped
        )
        chamfer_flipped = self._symmetric_chamfer_distance(real_aligned, sim_aligned_flipped)

        use_flipped = bool((chamfer_flipped < chamfer_direct).item())
        if use_flipped:
            sim_midpoint = sim_midpoint_flipped
            sim_rotation = sim_rotation_flipped
            sim_scale = sim_scale_flipped
            sim_aligned = sim_aligned_flipped
            baseline_chamfer = chamfer_flipped
        else:
            sim_midpoint = sim_midpoint_direct
            sim_rotation = sim_rotation_direct
            sim_scale = sim_scale_direct
            sim_aligned = sim_aligned_direct
            baseline_chamfer = chamfer_direct

        self._real_alignment_midpoint[env_id] = real_midpoint
        self._real_alignment_rotation[env_id] = real_rotation
        self._real_alignment_scale[env_id] = real_scale
        self._sim_alignment_midpoint[env_id] = sim_midpoint
        self._sim_alignment_rotation[env_id] = sim_rotation
        self._sim_alignment_scale[env_id] = sim_scale
        self._sim_alignment_flip_major[env_id] = use_flipped
        self._alignment_initialized[env_id] = True
        self._chamfer_baseline[env_id] = baseline_chamfer
        self._chamfer_baseline_valid[env_id] = True
        return real_aligned, sim_aligned, baseline_chamfer

    @staticmethod
    def _symmetric_chamfer_distance(points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
        points_a = torch.nan_to_num(points_a, nan=0.0, posinf=0.0, neginf=0.0)
        points_b = torch.nan_to_num(points_b, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise = torch.cdist(points_a.unsqueeze(0), points_b.unsqueeze(0), p=2).squeeze(0)
        pairwise = torch.nan_to_num(pairwise, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
        forward = pairwise.min(dim=1).values.mean()
        backward = pairwise.min(dim=0).values.mean()
        return torch.nan_to_num(0.5 * (forward + backward), nan=1.0e6, posinf=1.0e6, neginf=1.0e6)

    def _configure_gym_env_spaces(self):
        self._control_joint_ids = torch.arange(self.robot.num_joints, device=self.device)
        self._num_actions = self._control_joint_ids.shape[0]
        obs_dim = 13 + 2 * self._num_actions

        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)}
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.state_space = None
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_actions), device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.view(self.num_envs, self._num_actions)
        if not self._action_nan_reported:
            non_finite = ~torch.isfinite(actions)
            if non_finite.any():
                env_ids = torch.nonzero(non_finite.any(dim=1), as_tuple=False).flatten()
                env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
                bad_idx = torch.nonzero(non_finite[env_id], as_tuple=False).flatten().tolist()
                action_env = actions[env_id].detach().to("cpu").tolist()
                print(
                    f"[SalmonILEnv] non-finite action detected at step {self._action_log_counter} "
                    f"env{env_id} idx={bad_idx} -> {action_env}"
                )
                self._action_nan_reported = True
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions = actions
        self._torques = self.actions * self.cfg.action_scale
        warmup_mask = self._il_env_step < self._il_warmup_steps
        if warmup_mask.any():
            # During the initial settle window, keep the fish passive so it can fall
            # onto the ground before alignment, logging, and imitation begin.
            self._torques = self._torques.clone()
            self._torques[warmup_mask] = 0.0
        self._action_log_counter += 1
        if self.cfg.debug_logs:
            for env_id in range(self.num_envs):
                action_env = self.actions[env_id].detach().to("cpu")
                torque_env = self._torques[env_id].detach().to("cpu")
                print(
                    f"[SalmonILEnv] step {self._action_log_counter}: env{env_id} action "
                    f"(shape={tuple(self.actions.shape)}) -> {action_env.tolist()}"
                )
                print(
                    f"[SalmonILEnv] step {self._action_log_counter}: env{env_id} torque "
                    f"(shape={tuple(self._torques.shape)}) -> {torque_env.tolist()}"
                )
            self._pending_joint_log = True
            self._pending_joint_log_step = self._action_log_counter

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self._torques, joint_ids=self._control_joint_ids)

    def _get_observations(self) -> dict:
        if self.cfg.debug_logs and self._pending_joint_log:
            for env_id in range(self.num_envs):
                joint_pos_env = self.joint_pos[env_id].detach().to("cpu")
                print(
                    f"[SalmonILEnv] step {self._pending_joint_log_step}: env{env_id} joint_angle -> "
                    f"{joint_pos_env.tolist()}"
                )
            print("")
            self._pending_joint_log = False

        root_state = self.robot.data.root_state_w
        projected_gravity = self.robot.data.projected_gravity_b
        root_lin_vel = root_state[:, 7:10] * self.cfg.obs_scales.root_lin_vel
        root_ang_vel = root_state[:, 10:13] * self.cfg.obs_scales.root_ang_vel
        joint_pos_error = (self.joint_pos - self._default_joint_pos) * self.cfg.obs_scales.joint_pos
        joint_vel = self.joint_vel * self.cfg.obs_scales.joint_vel
        target_delta_xy = self.target_positions_w[:, 0:2] - root_state[:, 0:2]
        root_quat = root_state[:, 3:7]
        forward_b = torch.tensor([-1.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device).repeat(
            self.num_envs, 1
        )
        heading_vec = math_utils.quat_apply(root_quat, forward_b)
        heading_xy = heading_vec[:, 0:2]
        heading_xy = heading_xy / torch.linalg.norm(heading_xy, dim=1, keepdim=True).clamp(min=1.0e-6)
        distance_xy = torch.linalg.norm(target_delta_xy, dim=1)

        if self.cfg.debug_logs:
            for env_id in range(self.num_envs):
                dist_env = distance_xy[env_id].detach().to("cpu").item()
                print(f"[SalmonILEnv] step {self._action_log_counter}: env{env_id} distance_to_target -> {dist_env}")

        obs = torch.cat(
            (projected_gravity, root_lin_vel, root_ang_vel, joint_pos_error, joint_vel, heading_xy, target_delta_xy),
            dim=-1,
        )
        if not self._nan_reported and (~torch.isfinite(obs)).any():
            print("[SalmonILEnv] non-finite observation detected; clamping to finite values.")
            self._nan_reported = True
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        sim_pointclouds = self._get_live_sim_pointclouds_for_reward()
        real_frame_indices = self._get_real_frame_indices()
        current_chamfer = torch.empty(self.num_envs, dtype=torch.float32, device=self.device)
        # During the warmup window the fish is passive: no alignment, no logging,
        # and no imitation reward. Alignment starts from the first settled frame.
        warmup_mask = self._il_env_step < self._il_warmup_steps
        target_alignment_record = None
        for env_id in range(self.num_envs):
            if bool(warmup_mask[env_id].item()):
                current_chamfer[env_id] = 0.0
                continue
            real_points = self._real_pointcloud_sequence[real_frame_indices[env_id]]
            sim_points = sim_pointclouds[env_id]
            if bool(self._alignment_initialized[env_id].item()):
                real_aligned, sim_aligned = self._get_reward_aligned_pointclouds(env_id, real_points, sim_points)
                current_chamfer[env_id] = self._symmetric_chamfer_distance(real_aligned, sim_aligned)
            else:
                real_aligned, sim_aligned, baseline_chamfer = self._initialize_episode_alignment(env_id, real_points, sim_points)
                current_chamfer[env_id] = baseline_chamfer
            if (
                self._alignment_overlay_log_enabled
                and not self._alignment_overlay_log_saved
                and env_id == self._alignment_overlay_log_env_id
            ):
                target_alignment_record = (
                    env_id,
                    int(real_frame_indices[env_id].item()),
                    real_aligned,
                    sim_aligned,
                    current_chamfer[env_id],
                )

        if not self._reward_nan_reported and (~torch.isfinite(current_chamfer)).any():
            env_ids = torch.nonzero(~torch.isfinite(current_chamfer), as_tuple=False).flatten()
            env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
            print(
                f"[SalmonILEnv] non-finite Chamfer detected at step {self._action_log_counter} "
                f"env{env_id}; clamping to large finite value."
            )
            self._reward_nan_reported = True
        current_chamfer = torch.nan_to_num(current_chamfer, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
        if target_alignment_record is not None:
            self._maybe_record_alignment_overlay_frame(*target_alignment_record)

        # First-frame alignment: compute and store the real/sim centerline alignment once
        # at the beginning of each episode, then reuse the same transforms afterwards.
        # First-frame scale normalization: the stored transforms include the frame-0 body
        # length normalization, so later frames stay in that original canonical frame.
        # Baseline Chamfer storage: keep the frame-0 Chamfer once per environment for
        # diagnostics, even though the reward now depends directly on the current Chamfer.
        chamfer_excess = torch.clamp(current_chamfer - self._chamfer_baseline, min=0.0)
        chamfer_excess = torch.nan_to_num(chamfer_excess, nan=1.0e6, posinf=1.0e6, neginf=0.0)
        # Direct Chamfer shaping: lower current Chamfer yields larger reward.
        reward = self._chamfer_reward_weight / (1.0 + current_chamfer)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward[warmup_mask] = 0.0

        self._current_chamfer = current_chamfer
        post_warmup_mask = ~warmup_mask
        self._best_chamfer[post_warmup_mask] = torch.minimum(
            self._best_chamfer[post_warmup_mask], current_chamfer[post_warmup_mask]
        )
        self._episode_rewards += reward
        self._il_episode_frame += (~warmup_mask).to(dtype=self._il_episode_frame.dtype)
        self._il_env_step += 1

        cumulative_mean = self._episode_rewards.mean()
        self._cumulative_reward_queue.append(float(cumulative_mean.detach().to("cpu").item()))
        cumulative_smoothed = float(sum(self._cumulative_reward_queue) / len(self._cumulative_reward_queue))

        reward_mean = torch.nan_to_num(reward.mean(), nan=0.0, posinf=0.0, neginf=-1.0e6)
        chamfer_distance_mean = torch.nan_to_num(current_chamfer.mean(), nan=1.0e6, posinf=1.0e6, neginf=0.0)
        chamfer_baseline_mean = torch.nan_to_num(self._chamfer_baseline.mean(), nan=1.0e6, posinf=1.0e6, neginf=0.0)
        chamfer_excess_mean = torch.nan_to_num(chamfer_excess.mean(), nan=1.0e6, posinf=1.0e6, neginf=0.0)
        cumulative_mean = torch.nan_to_num(cumulative_mean, nan=0.0, posinf=0.0, neginf=-1.0e6)
        cumulative_smoothed = float(np.nan_to_num(cumulative_smoothed, nan=0.0, posinf=0.0, neginf=-1.0e6))

        if self.cfg.debug_logs:
            for env_id in range(self.num_envs):
                print(
                    f"[SalmonILEnv] step {self._action_log_counter}: env{env_id} rewards -> "
                    f"reward={reward[env_id].item():.6f}, "
                    f"chamfer={current_chamfer[env_id].item():.6f}, "
                    f"baseline={self._chamfer_baseline[env_id].item():.6f}, "
                    f"excess={chamfer_excess[env_id].item():.6f}, "
                    f"warmup={bool(warmup_mask[env_id].item())}"
                )

        self.extras["chamfer_distance"] = current_chamfer
        self.extras["chamfer_baseline"] = self._chamfer_baseline
        self.extras["chamfer_excess"] = chamfer_excess
        self.extras["il_warmup_active"] = warmup_mask
        self.extras["il_warmup_remaining_steps"] = torch.clamp(self._il_warmup_steps - self._il_env_step, min=0)
        self.extras["reward_total_mean"] = reward_mean
        self.extras["chamfer_distance_mean"] = chamfer_distance_mean
        self.extras["chamfer_baseline_mean"] = chamfer_baseline_mean
        self.extras["chamfer_excess_mean"] = chamfer_excess_mean
        self.extras["cumulative_reward_mean"] = cumulative_mean
        self.extras["cumulative_reward_mean_smoothed"] = cumulative_smoothed
        self.extras["log"] = {
            "reward_total_mean": reward_mean,
            "chamfer_distance_mean": chamfer_distance_mean,
            "chamfer_baseline_mean": chamfer_baseline_mean,
            "chamfer_excess_mean": chamfer_excess_mean,
            "cumulative_reward_mean": cumulative_mean,
            "cumulative_reward_mean_smoothed": cumulative_smoothed,
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self._maybe_capture_pointcloud()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_state = self.robot.data.root_state_w
        state_finite = torch.isfinite(root_state).all(dim=1)
        joint_pos_finite = torch.isfinite(self.joint_pos).all(dim=1)
        joint_vel_finite = torch.isfinite(self.joint_vel).all(dim=1)
        nonfinite_mask = ~(state_finite & joint_pos_finite & joint_vel_finite)
        if not self._state_nan_reported and nonfinite_mask.any():
            env_ids = torch.nonzero(nonfinite_mask, as_tuple=False).flatten()
            env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
            root_state_env = self.robot.data.root_state_w[env_id].detach().to("cpu").tolist()
            joint_pos_env = self.joint_pos[env_id].detach().to("cpu").tolist()
            joint_vel_env = self.joint_vel[env_id].detach().to("cpu").tolist()
            print(
                f"[SalmonILEnv] non-finite simulator state detected at step {self._action_log_counter} "
                f"env{env_id}; resetting env."
            )
            print(f"[SalmonILEnv] non-finite root_state env{env_id} -> {root_state_env}")
            print(f"[SalmonILEnv] non-finite joint_pos env{env_id} -> {joint_pos_env}")
            print(f"[SalmonILEnv] non-finite joint_vel env{env_id} -> {joint_vel_env}")
            self._state_nan_reported = True
        warmup_mask = self._il_env_step < self._il_warmup_steps
        local_root_pos = root_state[:, :3] - self.scene.env_origins
        root_position_blowup_mask = torch.linalg.norm(local_root_pos, dim=1) > float(self.cfg.root_position_limit)
        joint_position_blowup_mask = torch.amax(torch.abs(self.joint_pos), dim=1) > float(self.cfg.joint_position_limit)
        reward_pointcloud_invalid_mask = self._reward_pointcloud_invalid.clone()
        max_root_linear_speed = torch.linalg.norm(root_state[:, 7:10], dim=1)
        max_root_angular_speed = torch.linalg.norm(root_state[:, 10:13], dim=1)
        max_joint_abs_pos = torch.amax(torch.abs(self.joint_pos), dim=1)
        max_joint_abs_vel = torch.amax(torch.abs(self.joint_vel), dim=1)
        root_linear_speed_mask = (
            max_root_linear_speed > float(self.cfg.root_linear_velocity_limit)
        ) & ~warmup_mask
        root_angular_speed_mask = (
            max_root_angular_speed > float(self.cfg.root_angular_velocity_limit)
        ) & ~warmup_mask
        joint_speed_blowup_mask = (
            max_joint_abs_vel > float(self.cfg.joint_velocity_limit)
        ) & ~warmup_mask
        joint_position_blowup_mask = joint_position_blowup_mask & ~warmup_mask
        hard_blowup_mask = reward_pointcloud_invalid_mask | root_position_blowup_mask | joint_position_blowup_mask
        dynamic_blowup_mask = root_linear_speed_mask | root_angular_speed_mask | joint_speed_blowup_mask
        # Use a fixed rollout window for imitation learning: allow the fish to settle
        # during warmup, then keep the episode alive for a fixed duration so the agent
        # is judged on matching the video rather than tripping heuristic safety resets.
        fixed_reset_mask = self._il_env_step >= (self._il_warmup_steps + self._il_post_warmup_episode_steps)
        terminated = fixed_reset_mask | nonfinite_mask | hard_blowup_mask | dynamic_blowup_mask
        if self.cfg.debug_logs:
            for env_id in range(self.num_envs):
                if bool(terminated[env_id].item()) or bool(time_out[env_id].item()):
                    print(
                        f"[SalmonILEnv] done env{env_id} -> terminated={bool(terminated[env_id].item())}, "
                        f"timeout={bool(time_out[env_id].item())}, "
                        f"fixed_reset={bool(fixed_reset_mask[env_id].item())}, "
                        f"nonfinite={bool(nonfinite_mask[env_id].item())}, "
                        f"reward_pc_invalid={bool(reward_pointcloud_invalid_mask[env_id].item())}, "
                        f"root_pos_blowup={bool(root_position_blowup_mask[env_id].item())}, "
                        f"joint_pos_blowup={bool(joint_position_blowup_mask[env_id].item())}, "
                        f"root_lin_blowup={bool(root_linear_speed_mask[env_id].item())}, "
                        f"root_ang_blowup={bool(root_angular_speed_mask[env_id].item())}, "
                        f"joint_vel_blowup={bool(joint_speed_blowup_mask[env_id].item())}, "
                        f"max_joint_abs_pos={float(max_joint_abs_pos[env_id].item()):.6f}, "
                        f"max_joint_abs_vel={float(max_joint_abs_vel[env_id].item()):.6f}, "
                        f"max_root_lin_speed={float(max_root_linear_speed[env_id].item()):.6f}, "
                        f"max_root_ang_speed={float(max_root_angular_speed[env_id].item()):.6f}, "
                        f"il_env_step={int(self._il_env_step[env_id].item())}"
                    )

        done_envs = torch.nonzero(terminated | time_out, as_tuple=False).flatten()
        if len(done_envs) > 0:
            ep_rews = self._episode_rewards[done_envs]
            for reward in ep_rews.detach().to("cpu").tolist():
                self._episode_reward_queue.append(float(reward))
            queue_mean = float(sum(self._episode_reward_queue) / len(self._episode_reward_queue))
            best_chamfer = self._best_chamfer[done_envs].clone()
            unresolved_mask = ~torch.isfinite(best_chamfer)
            if unresolved_mask.any():
                best_chamfer[unresolved_mask] = self._current_chamfer[done_envs][unresolved_mask]
            best_chamfer = torch.nan_to_num(best_chamfer, nan=1.0e6, posinf=1.0e6, neginf=0.0)
            self.extras["last_episode_reward_mean"] = queue_mean
            self.extras["episode_reward_queue_mean"] = queue_mean
            self.extras["last_episode_best_chamfer_mean"] = best_chamfer.mean()
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if (
            self._alignment_overlay_log_enabled
            and not self._alignment_overlay_log_saved
            and self._alignment_overlay_records
            and (env_ids_tensor == self._alignment_overlay_log_env_id).any()
        ):
            self._flush_alignment_overlay_logs(mark_saved=False)
            self._alignment_overlay_records.clear()
        super()._reset_idx(env_ids)

        joint_pos = self._default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        joint_pos += sample_uniform(
            self.cfg.initial_joint_pos_range[0],
            self.cfg.initial_joint_pos_range[1],
            joint_pos.shape,
            joint_pos.device,
        )
        joint_vel += sample_uniform(
            self.cfg.initial_joint_vel_range[0],
            self.cfg.initial_joint_vel_range[1],
            joint_vel.shape,
            joint_vel.device,
        )

        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 7:13] = 0.0

        if not self.cfg.preserve_articulation_root_pose_on_reset:
            root_state[:, :3] = self.scene.env_origins[env_ids]
            pos_noise = sample_uniform(
                self.cfg.initial_root_pos_range[0],
                self.cfg.initial_root_pos_range[1],
                (len(env_ids), 2),
                joint_pos.device,
            )
            root_state[:, 0:2] += pos_noise
            root_state[:, 2] = self.cfg.target_root_height

            yaw_noise = sample_uniform(
                self.cfg.initial_root_rot_range[0],
                self.cfg.initial_root_rot_range[1],
                (len(env_ids), 1),
                joint_pos.device,
            ).squeeze(-1)
            yaw_quat = math_utils.quat_from_euler_xyz(
                torch.zeros_like(yaw_noise), torch.zeros_like(yaw_noise), yaw_noise
            )
            root_state[:, 3:7] = math_utils.quat_mul(yaw_quat, root_state[:, 3:7])

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.target_positions_w = self._compute_target_positions_world()

        self._episode_rewards[env_ids] = 0.0
        self._best_chamfer[env_ids] = float("inf")
        self._il_env_step[env_ids] = 0
        self._il_episode_frame[env_ids] = 0
        self._chamfer_baseline[env_ids] = 0.0
        self._chamfer_baseline_valid[env_ids] = False
        self._current_chamfer[env_ids] = 0.0
        self._alignment_initialized[env_ids] = False
        self._real_alignment_midpoint[env_ids] = 0.0
        self._real_alignment_rotation[env_ids] = torch.eye(
            3, dtype=self._real_alignment_rotation.dtype, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1, 1)
        self._real_alignment_scale[env_ids] = 1.0
        self._sim_alignment_midpoint[env_ids] = 0.0
        self._sim_alignment_rotation[env_ids] = torch.eye(
            3, dtype=self._sim_alignment_rotation.dtype, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1, 1)
        self._sim_alignment_scale[env_ids] = 1.0
        self._sim_alignment_flip_major[env_ids] = False
        # Reset per-episode diagnostics so later failures are still reported.
        self._nan_reported = False
        self._action_nan_reported = False
        self._action_range_reported = False
        self._torque_nan_reported = False
        self._reward_nan_reported = False
        self._joint_violation_reported = False
        self._pointcloud_nan_reported = False
        self._alignment_nan_reported = False
        self._state_nan_reported = False
        self._reward_pointcloud_invalid[env_ids] = False

        if self.cfg.real_pointcloud_random_start and self._real_pointcloud_frame_count > 1:
            start_indices = torch.randint(
                low=0,
                high=self._real_pointcloud_frame_count,
                size=(len(env_ids),),
                device=self.device,
            )
        else:
            start_frame = min(int(self.cfg.real_pointcloud_start_frame), self._real_pointcloud_frame_count - 1)
            start_indices = torch.full((len(env_ids),), start_frame, dtype=torch.long, device=self.device)
        self._il_real_frame_start[env_ids] = start_indices

    def close(self):
        if self._alignment_overlay_log_enabled and self._alignment_overlay_records and not self._alignment_overlay_log_saved:
            self._flush_alignment_overlay_logs(mark_saved=False)
        super().close()

    def _compute_target_positions_world(self) -> torch.Tensor:
        target_positions = self.scene.env_origins.clone()
        target_positions[:, 0:2] += self._target_offset_xy
        target_positions[:, 2] = self._target_height
        return target_positions
