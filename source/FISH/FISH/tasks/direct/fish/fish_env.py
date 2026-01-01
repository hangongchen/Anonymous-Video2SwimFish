# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
from pxr import Sdf, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .fish_env_cfg import FISH_GROUND_SIZE, FishEnvCfg


class FishEnv(DirectRLEnv):
    cfg: FishEnvCfg

    def __init__(self, cfg: FishEnvCfg, render_mode: str | None = None, **kwargs):
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
        # Episode statistics (per env).
        self._episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self._best_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._episode_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Track previous distance for progress-based reward.
        self._prev_distance = torch.zeros(self.num_envs, device=self.device)
        # Track initial distance and sparse-progress milestones.
        self._start_distance = torch.ones(self.num_envs, device=self.device)
        self._sparse_progress_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._sparse_progress_step = max(float(self.cfg.sparse_progress_step), 1.0e-6)
        self._sparse_progress_max_level = int(math.floor(1.0 / self._sparse_progress_step + 1.0e-6))
        # Rolling window for episode reward mean.
        self._episode_reward_queue = deque(maxlen=100)
        # Rolling window for cumulative reward mean smoothing.
        self._cumulative_reward_queue = deque(maxlen=100)
        # Track first non-finite observation to surface the root cause.
        self._nan_reported = False
        self._action_nan_reported = False
        self._action_range_reported = False
        self._torque_nan_reported = False
        self._reward_nan_reported = False
        self._joint_violation_reported = False

    def _setup_scene(self):
        # Keep stage units in meters for consistent world-scale configuration.
        UsdGeom.SetStageMetersPerUnit(self.scene.stage, 1)
        self.robot = Articulation(self.cfg.robot_cfg)
        # Force opaque preview materials to avoid accidental transparency from texture hookups.
        try:
            for prim in self.scene.stage.Traverse():
                if prim.GetTypeName() != "Shader":
                    continue
                shader = UsdShade.Shader(prim)
                shader_id = shader.GetIdAttr().Get()
                if shader_id != "UsdPreviewSurface":
                    continue
                opacity_input = shader.GetInput("opacity")
                if opacity_input:
                    attr = opacity_input.GetAttr()
                    if hasattr(attr, "SetConnections"):
                        # Author an empty connection list to block weaker layer connections.
                        attr.SetConnections([])
                    opacity_input.Set(1.0)
        except Exception as exc:
            if self.cfg.debug_logs:
                print(f"[FishEnv] opacity override failed ({exc})")
        if self.cfg.debug_logs:
            # Inspect material bindings and texture asset paths on the stage.
            try:
                materials = []
                assets = []
                bindings = []
                meshes = []
                for prim in self.scene.stage.Traverse():
                    if prim.IsA(UsdShade.Material):
                        materials.append(prim)
                    if prim.IsA(UsdGeom.Mesh):
                        img = UsdGeom.Imageable(prim)
                        visibility = img.ComputeVisibility()
                        purpose = img.GetPurposeAttr().Get()
                        meshes.append((prim, visibility, purpose))
                    if prim.GetTypeName() == "Shader":
                        shader = UsdShade.Shader(prim)
                        for inp in shader.GetInputs():
                            if inp.GetTypeName() == "asset":
                                val = inp.Get()
                                name = inp.GetBaseName()
                                if isinstance(val, Sdf.AssetPath):
                                    assets.append((prim.GetPath(), name, val.path, val.resolvedPath))
                                else:
                                    assets.append((prim.GetPath(), name, str(val), ""))
                material_shaders = []
                for mat in materials:
                    for child in mat.GetChildren():
                        if child.GetTypeName() == "Shader":
                            shader = UsdShade.Shader(child)
                            shader_id = shader.GetIdAttr().Get()
                            material_shaders.append((mat.GetPath(), child.GetPath(), shader_id))
                    api = UsdShade.MaterialBindingAPI(prim)
                    if api:
                        rel = api.GetDirectBindingRel()
                        if rel and rel.GetTargets():
                            bindings.append((prim.GetPath(), rel.GetTargets()))
                print(f"[FishEnv] stage materials -> {len(materials)}")
                for mat in materials[:10]:
                    print(f"[FishEnv] material -> {mat.GetPath()}")
                print(f"[FishEnv] material bindings -> {len(bindings)}")
                for prim_path, targets in bindings[:10]:
                    print(f"[FishEnv] binding -> {prim_path} -> {targets}")
                print(f"[FishEnv] meshes -> {len(meshes)}")
                for prim, visibility, purpose in meshes[:10]:
                    binding_api = UsdShade.MaterialBindingAPI(prim)
                    direct_rel = binding_api.GetDirectBindingRel()
                    direct_targets = direct_rel.GetTargets() if direct_rel else []
                    bound_material = binding_api.ComputeBoundMaterial()[0]
                    bound_path = bound_material.GetPath() if bound_material else None
                    uv_status = ""
                    if prim.GetPath().name == "Mesh":
                        primvars = UsdGeom.PrimvarsAPI(prim)
                        st = primvars.GetPrimvar("st")
                        if st and st.HasValue():
                            uv_status = f"st={st.GetInterpolation()}"
                        else:
                            uv_status = "st=missing"
                    print(
                        f"[FishEnv] mesh -> {prim.GetPath()} visibility={visibility} purpose={purpose} "
                        f"direct={direct_targets} bound={bound_path} {uv_status}".rstrip()
                    )
                print(f"[FishEnv] material shaders -> {len(material_shaders)}")
                for mat_path, shader_path, shader_id in material_shaders[:10]:
                    print(f"[FishEnv] shader -> {mat_path} {shader_path} id={shader_id}")
                for mat_path, shader_path, shader_id in material_shaders:
                    if shader_id != "UsdPreviewSurface":
                        continue
                    shader = UsdShade.Shader(self.scene.stage.GetPrimAtPath(shader_path))
                    for input_name in ("diffuseColor", "opacity", "metallic", "roughness", "normal"):
                        inp = shader.GetInput(input_name)
                        if not inp:
                            continue
                        connections = inp.GetConnectedSource()
                        if connections:
                            src = connections[0].GetPath() if connections else None
                            print(f"[FishEnv] preview input -> {shader_path}.{input_name} connected={src}")
                        else:
                            print(f"[FishEnv] preview input -> {shader_path}.{input_name} value={inp.Get()}")
                print(f"[FishEnv] texture assets -> {len(assets)}")
                for prim_path, name, path_val, resolved in assets[:20]:
                    print(f"[FishEnv] asset -> {prim_path} input={name} path={path_val} resolved={resolved}")
            except Exception as exc:
                print(f"[FishEnv] material inspection failed ({exc})")
        if self.cfg.debug_logs:
            try:
                print(f"[FishEnv] joints -> {self.robot.joint_names} (num_joints={self.robot.num_joints})")
                limits = self.robot.data.soft_joint_pos_limits[0].detach().to("cpu").tolist()
                print(f"[FishEnv] joint soft limits (env0) -> {limits}")
            except Exception as exc:
                print(f"[FishEnv] joint debug log unavailable ({exc})")
        ground_cfg = GroundPlaneCfg()
        ground_cfg.size = FISH_GROUND_SIZE
        env0_ground_path = f"{self.scene.env_ns}/env_0/ground"
        spawn_ground_plane(prim_path=env0_ground_path, cfg=ground_cfg)
        target_translation = (
            float(self.cfg.target_offset_xy[0]),
            float(self.cfg.target_offset_xy[1]),
            float(self.cfg.target_marker_height),
        )
        target_prim_path = f"{self.scene.env_ns}/env_0/target_marker"
        self.cfg.target_marker_cfg.func(
            prim_path=target_prim_path,
            cfg=self.cfg.target_marker_cfg,
            translation=target_translation,
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu" or not self.cfg.scene.replicate_physics:
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _configure_gym_env_spaces(self):
        self._control_joint_ids = torch.arange(self.robot.num_joints, device=self.device)
        self._num_actions = self._control_joint_ids.shape[0]
        obs_dim = 13 + 2 * self._num_actions

        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            }
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.state_space = None

        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self._num_actions,), dtype=np.float32
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_actions), device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # ensure actions always include the env dimension so we drive all joints
        actions = actions.view(self.num_envs, self._num_actions)
        if not self._action_nan_reported:
            non_finite = ~torch.isfinite(actions)
            if non_finite.any():
                env_ids = torch.nonzero(non_finite.any(dim=1), as_tuple=False).flatten()
                env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
                bad_idx = torch.nonzero(non_finite[env_id], as_tuple=False).flatten().tolist()
                action_env = actions[env_id].detach().to("cpu").tolist()
                print(
                    f"[FishEnv] non-finite action detected at step {self._action_log_counter} "
                    f"env{env_id} idx={bad_idx} -> {action_env}"
                )
                self._action_nan_reported = True
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        if not self._action_range_reported:
            over_limit = actions.abs() > 1.0
            if over_limit.any():
                env_ids = torch.nonzero(over_limit.any(dim=1), as_tuple=False).flatten()
                env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
                count = int(over_limit.sum().item())
                total = int(actions.numel())
                max_abs = float(actions.abs().max().item())
                action_env = actions[env_id].detach().to("cpu").tolist()
                print(
                    f"[FishEnv] action out of [-1, 1] at step {self._action_log_counter} "
                    f"{count}/{total} max_abs={max_abs} env{env_id} -> {action_env}"
                )
                self._action_range_reported = True
        self.actions = actions
        self._torques = self.actions * self.cfg.action_scale
        if not self._torque_nan_reported:
            non_finite_torque = ~torch.isfinite(self._torques)
            if non_finite_torque.any():
                env_ids = torch.nonzero(non_finite_torque.any(dim=1), as_tuple=False).flatten()
                env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
                bad_idx = torch.nonzero(non_finite_torque[env_id], as_tuple=False).flatten().tolist()
                torque_env = self._torques[env_id].detach().to("cpu").tolist()
                print(
                    f"[FishEnv] non-finite torque detected at step {self._action_log_counter} "
                    f"env{env_id} idx={bad_idx} -> {torque_env}"
                )
                self._torque_nan_reported = True
        self._action_log_counter += 1
        if self.cfg.debug_logs:
            for env_id in range(self.num_envs):
                action_env = self.actions[env_id].detach().to("cpu")
                torque_env = self._torques[env_id].detach().to("cpu")
                print(
                    f"[FishEnv] step {self._action_log_counter}: env{env_id} action "
                    f"(shape={tuple(self.actions.shape)}) -> {action_env.tolist()}"
                )
                print(
                    f"[FishEnv] step {self._action_log_counter}: env{env_id} torque "
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
                    f"[FishEnv] step {self._pending_joint_log_step}: env{env_id} joint_angle -> "
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
        # Fish asset faces -X in its body frame.
        forward_b = torch.tensor([-1.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
        forward_b = forward_b.repeat(self.num_envs, 1)
        heading_vec = math_utils.quat_apply(root_quat, forward_b)
        heading_xy = heading_vec[:, 0:2]
        heading_xy_norm = torch.linalg.norm(heading_xy, dim=1, keepdim=True).clamp(min=1.0e-6)
        heading_dir_xy = heading_xy / heading_xy_norm

        distance_xy = torch.linalg.norm(target_delta_xy, dim=1)

        if self.cfg.debug_logs:
            for env_id in range(self.num_envs):
                root_pos_env = root_state[env_id, 0:3].detach().to("cpu")
                root_lin_env = root_lin_vel[env_id].detach().to("cpu")
                root_ang_env = root_ang_vel[env_id].detach().to("cpu")
                joint_pos_err_env = joint_pos_error[env_id].detach().to("cpu")
                dist_env = distance_xy[env_id].detach().to("cpu").item()
                # print(f"[FishEnv] step {self._action_log_counter}: env{env_id} root_lin_vel -> {root_lin_env.tolist()}")
                # print(f"[FishEnv] step {self._action_log_counter}: env{env_id} root_ang_vel -> {root_ang_env.tolist()}")
                # print(f"[FishEnv] step {self._action_log_counter}: env{env_id} joint_pos_error -> {joint_pos_err_env.tolist()}")
                # print(f"[FishEnv] step {self._action_log_counter}: env{env_id} root_pos -> {root_pos_env.tolist()}")
                print(f"[FishEnv] step {self._action_log_counter}: env{env_id} distance_to_target -> {dist_env}")

        obs = torch.cat(
            (
                projected_gravity,
                root_lin_vel,
                root_ang_vel,
                joint_pos_error,
                joint_vel,
                heading_dir_xy,
                target_delta_xy,
            ),
            dim=-1,
        )
        if not self._nan_reported:
            non_finite = ~torch.isfinite(obs)
            if non_finite.any():
                env_ids = torch.nonzero(non_finite.any(dim=1), as_tuple=False).flatten()
                env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
                bad_idx = torch.nonzero(non_finite[env_id], as_tuple=False).flatten().tolist()
                root_state_env = root_state[env_id].detach().to("cpu").tolist()
                projected_gravity_env = projected_gravity[env_id].detach().to("cpu").tolist()
                joint_pos_env = self.joint_pos[env_id].detach().to("cpu").tolist()
                joint_vel_env = self.joint_vel[env_id].detach().to("cpu").tolist()
                target_delta_env = target_delta_xy[env_id].detach().to("cpu").tolist()
                obs_env = obs[env_id].detach().to("cpu").tolist()
                print(
                    f"[FishEnv] non-finite obs detected at step {self._action_log_counter} "
                    f"env{env_id} idx={bad_idx}"
                )
                print(f"[FishEnv] non-finite root_state env{env_id} -> {root_state_env}")
                print(f"[FishEnv] non-finite projected_gravity env{env_id} -> {projected_gravity_env}")
                print(f"[FishEnv] non-finite joint_pos env{env_id} -> {joint_pos_env}")
                print(f"[FishEnv] non-finite joint_vel env{env_id} -> {joint_vel_env}")
                print(f"[FishEnv] non-finite target_delta_xy env{env_id} -> {target_delta_env}")
                print(f"[FishEnv] non-finite obs env{env_id} -> {obs_env}")
                self._nan_reported = True
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_state = self.robot.data.root_state_w
        distance_xy = torch.linalg.norm(root_state[:, 0:2] - self.target_positions_w[:, 0:2], dim=1)
        target_delta_xy = self.target_positions_w[:, 0:2] - root_state[:, 0:2]
        target_delta = self.target_positions_w - root_state[:, 0:3]
        if not self._reward_nan_reported:
            non_finite_root = ~torch.isfinite(root_state)
            non_finite_target = ~torch.isfinite(self.target_positions_w)
            non_finite_distance = ~torch.isfinite(distance_xy)
            bad_envs = non_finite_root.any(dim=1) | non_finite_target.any(dim=1) | non_finite_distance
            if bad_envs.any():
                env_ids = torch.nonzero(bad_envs, as_tuple=False).flatten()
                env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
                root_state_env = root_state[env_id].detach().to("cpu").tolist()
                target_env = self.target_positions_w[env_id].detach().to("cpu").tolist()
                target_delta_env = target_delta[env_id].detach().to("cpu").tolist()
                distance_env = distance_xy[env_id].detach().to("cpu").item()
                print(
                    f"[FishEnv] non-finite reward inputs at step {self._action_log_counter} env{env_id} "
                    f"distance={distance_env} target={target_env} target_delta={target_delta_env} "
                    f"root_state={root_state_env}"
                )
                self._reward_nan_reported = True
        safe_target_delta_xy = torch.nan_to_num(target_delta_xy, nan=0.0, posinf=0.0, neginf=0.0)
        safe_target_delta = torch.nan_to_num(target_delta, nan=0.0, posinf=0.0, neginf=0.0)

        finite_mask = torch.isfinite(distance_xy)
        safe_distance_xy = torch.where(finite_mask, distance_xy, torch.zeros_like(distance_xy))
        success = (safe_distance_xy < self.cfg.success_radius) & finite_mask
        success_rate = success.float().mean()

        safe_prev_distance = torch.nan_to_num(self._prev_distance, nan=0.0, posinf=0.0, neginf=0.0)
        progress_reward = (safe_prev_distance - safe_distance_xy) * self.cfg.rew_scales.distance
        progress_reward = torch.nan_to_num(progress_reward, nan=0.0, posinf=0.0, neginf=0.0)
        progress_reward = progress_reward * finite_mask.float()
        safe_start_distance = torch.nan_to_num(self._start_distance, nan=0.0, posinf=0.0, neginf=0.0)
        progress_frac = (safe_start_distance - safe_distance_xy) / safe_start_distance.clamp(min=1.0e-6)
        progress_frac = torch.nan_to_num(progress_frac, nan=0.0, posinf=0.0, neginf=0.0)
        progress_frac = torch.clamp(progress_frac, min=0.0, max=1.0)
        progress_level = torch.floor(progress_frac / self._sparse_progress_step).to(torch.long)
        progress_level = torch.clamp(progress_level, min=0, max=self._sparse_progress_max_level)
        level_delta = torch.clamp(progress_level - self._sparse_progress_level, min=0)
        sparse_progress_reward = level_delta.float() * self.cfg.rew_scales.sparse_progress
        sparse_progress_reward = sparse_progress_reward * finite_mask.float()
        self._sparse_progress_level = torch.where(
            finite_mask,
            torch.maximum(self._sparse_progress_level, progress_level),
            self._sparse_progress_level,
        )
        success_reward = success.float() * self.cfg.rew_scales.success
        success_reward = torch.nan_to_num(success_reward, nan=0.0, posinf=0.0, neginf=0.0)
        time_reward = -torch.ones_like(progress_reward) * self.cfg.rew_scales.time
        time_reward = time_reward * finite_mask.float()
        root_quat = torch.nan_to_num(root_state[:, 3:7], nan=0.0, posinf=0.0, neginf=0.0)
        # Fish asset faces -X in its body frame.
        forward_b = torch.tensor([-1.0, 0.0, 0.0], dtype=root_state.dtype, device=root_state.device)
        forward_b = forward_b.repeat(self.num_envs, 1)
        heading_vec = math_utils.quat_apply(root_quat, forward_b)
        heading_xy = heading_vec[:, 0:2]
        heading_xy_norm = torch.linalg.norm(heading_xy, dim=1, keepdim=True).clamp(min=1.0e-6)
        heading_dir_xy = heading_xy / heading_xy_norm
        target_dir_xy = safe_target_delta_xy / safe_distance_xy.clamp(min=1.0e-6).unsqueeze(-1)
        heading_proj = (heading_dir_xy * target_dir_xy).sum(dim=1)
        heading_reward = heading_proj * self.cfg.rew_scales.heading
        heading_reward = torch.nan_to_num(heading_reward, nan=0.0, posinf=0.0, neginf=0.0)
        heading_reward = heading_reward * finite_mask.float()
        target_dir = safe_target_delta / torch.linalg.norm(safe_target_delta, dim=1, keepdim=True).clamp(min=1.0e-6)
        orientation_proj = (heading_vec * target_dir).sum(dim=1)
        orientation_reward = orientation_proj * self.cfg.rew_scales.orientation
        orientation_reward = torch.nan_to_num(orientation_reward, nan=0.0, posinf=0.0, neginf=0.0)
        orientation_reward = orientation_reward * finite_mask.float()
        total_reward = (
            progress_reward
            + sparse_progress_reward
            + success_reward
            + heading_reward
            + orientation_reward
            + time_reward
        )
        self._prev_distance = safe_distance_xy.detach()
        if self.cfg.debug_logs:
            for env_id in range(self.num_envs):
                print(
                    f"[FishEnv] step {self._action_log_counter}: env{env_id} rewards -> "
                    f"progress={progress_reward[env_id].item():.6f}, "
                    f"sparse_progress={sparse_progress_reward[env_id].item():.6f}, "
                    f"success={success_reward[env_id].item():.6f}, "
                    f"heading={heading_reward[env_id].item():.6f}, "
                    f"orientation={orientation_reward[env_id].item():.6f}, "
                    f"time={time_reward[env_id].item():.6f}, "
                    f"total={total_reward[env_id].item():.6f}, "
                    f"distance_xy={safe_distance_xy[env_id].item():.6f}, "
                    f"success={bool(success[env_id].item())}"
                )

        self.extras["distance_to_target"] = safe_distance_xy
        self.extras["success"] = success
        self.extras["success_rate"] = success_rate
        self.extras["reward_progress_mean"] = progress_reward.mean()
        self.extras["reward_sparse_progress_mean"] = sparse_progress_reward.mean()
        self.extras["reward_heading_mean"] = heading_reward.mean()
        self.extras["reward_orientation_mean"] = orientation_reward.mean()
        self.extras["reward_time_mean"] = time_reward.mean()
        self.extras["reward_success_mean"] = success_reward.mean()
        self.extras["reward_total_mean"] = total_reward.mean()
        cumulative_mean = self._episode_rewards.mean()
        self._cumulative_reward_queue.append(float(cumulative_mean.detach().to("cpu").item()))
        if self._cumulative_reward_queue:
            cumulative_smoothed = float(sum(self._cumulative_reward_queue) / len(self._cumulative_reward_queue))
        else:
            cumulative_smoothed = 0.0
        self.extras["cumulative_reward_mean"] = cumulative_mean
        self.extras["cumulative_reward_mean_smoothed"] = cumulative_smoothed
        # Provide scalar episode-style logs for RL-Games/W&B.
        self.extras["log"] = {
            "distance_to_target_mean": safe_distance_xy.mean(),
            "success_rate": success_rate,
            "reward_progress_mean": progress_reward.mean(),
            "reward_sparse_progress_mean": sparse_progress_reward.mean(),
            "reward_heading_mean": heading_reward.mean(),
            "reward_orientation_mean": orientation_reward.mean(),
            "reward_time_mean": time_reward.mean(),
            "reward_success_mean": success_reward.mean(),
            "reward_total_mean": total_reward.mean(),
            "cumulative_reward_mean": cumulative_mean,
            "cumulative_reward_mean_smoothed": cumulative_smoothed,
        }
        # Update episode stats.
        self._episode_rewards += total_reward
        self._best_distance = torch.minimum(self._best_distance, safe_distance_xy)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_height = self.robot.data.root_state_w[:, 2]
        fallen = root_height < self.cfg.termination_height
        joint_limits = self._soft_joint_limits
        joint_violation = torch.any(
            (self.joint_pos < (joint_limits[:, :, 0] - self.cfg.joint_limit_margin))
            | (self.joint_pos > (joint_limits[:, :, 1] + self.cfg.joint_limit_margin)),
            dim=1,
        )
        if not self._joint_violation_reported and joint_violation.any():
            env_ids = torch.nonzero(joint_violation, as_tuple=False).flatten()
            env_id = int(env_ids[0].item()) if env_ids.numel() > 0 else 0
            limit_low = joint_limits[env_id, :, 0] - self.cfg.joint_limit_margin
            limit_high = joint_limits[env_id, :, 1] + self.cfg.joint_limit_margin
            bad_joint_mask = (self.joint_pos[env_id] < limit_low) | (self.joint_pos[env_id] > limit_high)
            bad_joints = torch.nonzero(bad_joint_mask, as_tuple=False).flatten().tolist()
            joint_pos_env = self.joint_pos[env_id].detach().to("cpu")
            limit_low_env = limit_low.detach().to("cpu")
            limit_high_env = limit_high.detach().to("cpu")
            pos_vals = [float(joint_pos_env[i].item()) for i in bad_joints]
            low_vals = [float(limit_low_env[i].item()) for i in bad_joints]
            high_vals = [float(limit_high_env[i].item()) for i in bad_joints]
            print(
                f"[FishEnv] joint limit violation at step {self._action_log_counter} env{env_id} "
                f"joints={bad_joints} pos={pos_vals} low={low_vals} high={high_vals}"
            )
            self._joint_violation_reported = True
        distance_xy = torch.linalg.norm(self.robot.data.root_state_w[:, 0:2] - self.target_positions_w[:, 0:2], dim=1)
        success = distance_xy < self.cfg.success_radius
        state_finite = torch.isfinite(self.robot.data.root_state_w).all(dim=1)
        joint_pos_finite = torch.isfinite(self.joint_pos).all(dim=1)
        joint_vel_finite = torch.isfinite(self.joint_vel).all(dim=1)
        nonfinite_mask = ~(state_finite & joint_pos_finite & joint_vel_finite)
        terminated = fallen | joint_violation | success | nonfinite_mask
        if self.cfg.debug_logs:
            done_env0 = terminated[0].item()
            timeout_env0 = time_out[0].item()
            success_env0 = success[0].item()
            if nonfinite_mask[0].item():
                print("[FishEnv] done env0 -> non-finite state detected; resetting env.")
            if done_env0 or timeout_env0:
                print(
                    f"[FishEnv] done env0 -> terminated={done_env0}, timeout={timeout_env0}, "
                    f"fallen={fallen[0].item()}, joint_limit_violation={joint_violation[0].item()}, "
                    f"success={success_env0}"
                )
                if success_env0:
                    print("[FishEnv] success env0 -> reached target radius")
        done_envs = torch.nonzero(terminated | time_out, as_tuple=False).flatten()
        if len(done_envs) > 0:
            # Episode-level stats for logging (wandb/weave via RL-Games observer).
            ep_rews = self._episode_rewards[done_envs]
            ep_success = success[done_envs].float()
            for reward in ep_rews.detach().to("cpu").tolist():
                self._episode_reward_queue.append(float(reward))
            if self._episode_reward_queue:
                queue_mean = float(sum(self._episode_reward_queue) / len(self._episode_reward_queue))
            else:
                queue_mean = 0.0
            self.extras["last_episode_reward_mean"] = queue_mean
            self.extras["episode_reward_queue_mean"] = queue_mean
            self.extras["last_episode_success_rate"] = ep_success.mean()
            self.extras["success_rate_episode"] = ep_success.mean()
            self.extras["last_episode_best_distance_mean"] = self._best_distance[done_envs].mean()
            if not self.cfg.debug_logs:
                for env_id in done_envs.tolist():
                    self._episode_counts[env_id] += 1
                    best_dist = self._best_distance[env_id].item()
                    cum_rew = self._episode_rewards[env_id].item()
                    success_env = success[env_id].item()
                    print(
                        f"[FishEnv] episode end env{env_id} -> "
                        f"success={success_env}, cum_reward={cum_rew:.4f}, "
                        f"best_distance={best_dist:.4f}, total_episodes={self._episode_counts[env_id].item()}"
                    )
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if 0 in env_ids:
            # Log whenever environment 0 is reset for easier debugging.
            print(f"[FishEnv] reset env0 at step {self._action_log_counter}")

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
        root_state[:, :3] = self.scene.env_origins[env_ids]
        pos_noise = sample_uniform(
            self.cfg.initial_root_pos_range[0],
            self.cfg.initial_root_pos_range[1],
            (len(env_ids), 2),
            joint_pos.device,
        )
        root_state[:, 0:2] += pos_noise
        root_state[:, 2] = self.cfg.target_root_height
        root_state[:, 7:13] = 0.0

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
        # Reset episode stats for these environments.
        self._episode_rewards[env_ids] = 0.0
        self._best_distance[env_ids] = float("inf")
        distance_xy = torch.linalg.norm(
            self.robot.data.root_state_w[env_ids, 0:2] - self.target_positions_w[env_ids, 0:2],
            dim=1,
        )
        distance_xy = torch.nan_to_num(distance_xy, nan=0.0, posinf=0.0, neginf=0.0)
        self._prev_distance[env_ids] = distance_xy
        self._start_distance[env_ids] = distance_xy
        self._sparse_progress_level[env_ids] = 0

    def _compute_target_positions_world(self) -> torch.Tensor:
        target_positions = self.scene.env_origins.clone()
        target_positions[:, 0:2] += self._target_offset_xy
        target_positions[:, 2] = self._target_height
        return target_positions
