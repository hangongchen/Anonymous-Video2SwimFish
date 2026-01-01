# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Salmon imitation-learning task in ANALYTIC water.

`SalmonILWaterEnv(SalmonSwimEnv)` inherits the proven-stable analytic-water + FEM
physics of the swim task (zero-gravity MuJoCo ellipsoid hydro, per-slice bone
ellipsoids, PD-position D6 control, the dt = 1/960 FEM-CFL-stable integration, the
FEM material create+bind cure, and the `_soft_view` DeformablePrim over the
deformable mesh). It OVERRIDES only the task layer:

  * reward  : symmetric Chamfer distance between the simulated FEM-fish point cloud
              (read from the deformable simulation mesh via the inherited `_soft_view`)
              and a real-fish video point cloud frame, compared in a per-episode
              centerline-canonical frame (pose/scale normalized). reward = w/(1+chamfer).
  * obs     : proprioception + a 2-D imitation phase (where in the real clip we are).
  * dones   : timeout + the inherited blow-up guards (NO swim "reached target" success).
  * reset   : the inherited robot/soft-body reset, plus the per-env IL frame pointers.

The Chamfer alignment machinery (centerline extraction, alignment transform, flip
disambiguation, symmetric chamfer) is ported verbatim from the original
`salmon_IL.py` -- it is physics-independent and reusable as-is.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils

from .salmon_swim_env import SalmonSwimEnv
from .salmon_IL_water_cfg import SalmonILWaterEnvCfg


class SalmonILWaterEnv(SalmonSwimEnv):
    cfg: SalmonILWaterEnvCfg

    def __init__(self, cfg: SalmonILWaterEnvCfg, render_mode: str | None = None, **kwargs):
        # Builds the scene, the articulation + analytic hydro, the FEM material, AND
        # (because cfg.reset_deformable & cfg.with_deformable) the `_soft_view`
        # DeformablePrim we reuse to read the deformable mesh for the reward.
        super().__init__(cfg, render_mode, **kwargs)
        self._il_setup()
        self._monitor_setup()

    # ------------------------------------------------------------------ IL setup ----
    def _il_setup(self) -> None:
        if getattr(self, "_soft_view", None) is None or getattr(self, "_n_soft_nodes", 0) < 1:
            raise RuntimeError(
                "[SalmonILWater] the deformable `_soft_view` was not created -- the imitation "
                "reward needs the FEM mesh. Ensure cfg.with_deformable=True and cfg.reset_deformable=True."
            )

        self._il_point_count = int(self.cfg.chamfer_point_count)
        self._chamfer_reward_weight = float(self.cfg.chamfer_reward_weight)

        # warmup window in CONTROL steps (step_dt = decimation * SIM_DT = 1/30 s)
        step_dt = self.cfg.decimation * self.cfg.sim.dt
        self._il_warmup_steps = int(round(float(self.cfg.il_warmup_seconds) / max(step_dt, 1e-9)))

        # real-fish target sequence -> (num_frames, point_count, 3)
        self._load_real_pointcloud_sequence()

        # ---- sim-fish point cloud source ----
        # PREFER the deformable SURFACE: embed the dense render-surface rest points into the
        # simulation tet mesh (barycentric weights), so each step we reconstruct the deformed
        # fish SURFACE -- a proper fish shape that matches the real surface cloud in both
        # geometry and (centerline) scale. Fall back to the raw simulation nodes (the coarse
        # tetrahedral lattice) only if the embedding cannot be built.
        self._sim_surface_node_indices = None   # (P,4) long
        self._sim_surface_weights = None         # (P,4) float
        self._sim_reward_sample_indices = None   # (P,) long  (fallback)
        node_idx = weights = None
        prim = self._find_env0_deformable_prim()
        if prim is not None:
            node_idx, weights = self._build_deformable_surface_embedding(prim)
        if node_idx is not None and weights is not None:
            ids = self._uniform_sample_indices(node_idx.shape[0], self._il_point_count)
            self._sim_surface_node_indices = torch.as_tensor(node_idx[ids], dtype=torch.long, device=self.device)
            self._sim_surface_weights = torch.as_tensor(weights[ids], dtype=torch.float32, device=self.device)
            self._sim_src = f"surface-embedded {self._sim_surface_node_indices.shape[0]} pts (from {node_idx.shape[0]} surf)"
        else:
            ids = self._uniform_sample_indices(self._n_soft_nodes, self._il_point_count)
            self._sim_reward_sample_indices = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            self._sim_src = f"raw sim-nodes {len(ids)} pts (surface embedding unavailable)"

        n = self.num_envs
        dev = self.device
        # per-env IL pointers
        self._il_env_step = torch.zeros(n, dtype=torch.long, device=dev)
        self._il_episode_frame = torch.zeros(n, dtype=torch.long, device=dev)
        self._il_real_frame_start = torch.full((n,), int(self.cfg.real_pointcloud_start_frame),
                                               dtype=torch.long, device=dev)
        # per-env, per-episode alignment transforms (real & sim each get their own)
        self._real_alignment_midpoint = torch.zeros(n, 3, device=dev)
        self._real_alignment_rotation = torch.eye(3, device=dev).unsqueeze(0).repeat(n, 1, 1)
        self._real_alignment_scale = torch.ones(n, device=dev)
        self._sim_alignment_midpoint = torch.zeros(n, 3, device=dev)
        self._sim_alignment_rotation = torch.eye(3, device=dev).unsqueeze(0).repeat(n, 1, 1)
        self._sim_alignment_scale = torch.ones(n, device=dev)
        self._sim_alignment_flip_major = torch.zeros(n, dtype=torch.bool, device=dev)
        self._alignment_initialized = torch.zeros(n, dtype=torch.bool, device=dev)
        # previous-frame REAL rotation per env, to keep the alignment signs temporally consistent
        # (resolve the SVD head-tail / up-down sign ambiguity -> no frame-to-frame flips).
        self._prev_real_rot = torch.eye(3, device=dev).unsqueeze(0).repeat(n, 1, 1)
        self._prev_real_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        # baseline = MEAN of the first `baseline_num_frames` post-warmup chamfers (per env), to
        # de-noise the single-frame baseline. Accumulated until ready, then frozen for the episode.
        self._baseline_num_frames = max(1, int(self.cfg.baseline_num_frames))
        self._baseline_sum = torch.zeros(n, device=dev)
        self._baseline_n = torch.zeros(n, dtype=torch.long, device=dev)
        self._baseline_ready = torch.zeros(n, dtype=torch.bool, device=dev)
        self._chamfer_baseline = torch.zeros(n, device=dev)
        self._chamfer_baseline_valid = torch.zeros(n, dtype=torch.bool, device=dev)
        self._best_chamfer = torch.full((n,), float("inf"), device=dev)
        self._current_chamfer = torch.zeros(n, device=dev)
        self._reward_pointcloud_invalid = torch.zeros(n, dtype=torch.bool, device=dev)
        # one-shot non-finite reports
        self._pointcloud_nan_reported = False
        self._alignment_nan_reported = False
        self._reward_nan_reported = False
        self._il_step_counter = 0

        # ---- DEMO-imitation mode: match a recorded RL rollout in WORLD frame (no alignment) ----
        self._demo_mode = bool(getattr(self.cfg, "demo_path", ""))
        if self._demo_mode:
            dd = np.load(self.cfg.demo_path)
            self._demo_pc = torch.as_tensor(dd["pc"], dtype=torch.float32, device=self.device)   # (T,N,3) world
            self._demo_root_np = np.asarray(dd["root"][:, :3], dtype=np.float32) if "root" in dd else None  # GT root (env-local)
            # full state for Reference State Initialization (RSI); auto-disable if the demo lacks it
            self._demo_has_full = all(k in dd for k in ("root", "joint_pos", "joint_vel", "nodal_vel"))
            if self._demo_has_full:
                self._demo_root_t = torch.as_tensor(dd["root"], dtype=torch.float32, device=self.device)      # (T,13) env0-local
                self._demo_jpos = torch.as_tensor(dd["joint_pos"], dtype=torch.float32, device=self.device)   # (T,nj)
                self._demo_jvel = torch.as_tensor(dd["joint_vel"], dtype=torch.float32, device=self.device)   # (T,nj)
                self._demo_nvel = torch.as_tensor(dd["nodal_vel"], dtype=torch.float32, device=self.device)   # (T,N,3)
            self._demo_rsi = bool(getattr(self.cfg, "demo_rsi", False)) and self._demo_has_full
            self._demo_start_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._demo_last_centroid = torch.zeros(self.num_envs, device=self.device)
            # velocity + head/tail-axis (angular vel) state. head/tail labelled lazily from skeleton.
            self._dt_ctrl = float(self.cfg.decimation * self.cfg.sim.dt)
            self._head_mask = None                # (N,) bool, computed on first reward step
            self._demo_cvel = None                # (T,3) demo centroid velocity (precomputed)
            self._demo_avel = None                # (T,3) demo head->tail axis velocity
            self._demo_axis_seq = None            # (T,3) demo head->tail unit axis per frame
            self._prev_sim_centroid = torch.zeros(self.num_envs, 3, device=self.device)
            self._prev_sim_axis = torch.zeros(self.num_envs, 3, device=self.device)
            self._prev_vel_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            print(f"[SalmonILWater] RSI={'ON' if self._demo_rsi else 'off'} (full state: {self._demo_has_full}), "
                  f"ET centroid>{self.cfg.demo_et_centroid}m", flush=True)
            self._demo_T = int(self._demo_pc.shape[0])
            # use the RAW FEM nodal positions (world), ALL nodes -> same representation as the demo.
            self._sim_surface_node_indices = None
            self._sim_reward_sample_indices = torch.arange(self._n_soft_nodes, device=self.device)
            # episode == demo length (enforced in _get_dones; max_episode_length is a read-only property)
            print(f"[SalmonILWater] DEMO mode: {self._demo_T} frames x {self._demo_pc.shape[1]} world pts, "
                  f"reward={self.cfg.demo_reward_kind}(world_chamfer/{self.cfg.demo_reward_temp}) "
                  f"- {self.cfg.demo_far_weight}*centroid; episode={self._demo_T} steps", flush=True)

        print(f"[SalmonILWater] IL ready: {self._real_pointcloud_frame_count} real frames "
              f"(loop={self.cfg.real_pointcloud_loop}), {self._il_point_count} chamfer pts, "
              f"sim={self._sim_src}, warmup {self._il_warmup_steps} steps, "
              f"reward=w*per-frame improvement below baseline(mean of first {self._baseline_num_frames} "
              f"frames, relative={self.cfg.chamfer_reward_relative}) w={self._chamfer_reward_weight}", flush=True)

    # -------------------------------------------- deformable SURFACE embedding ----
    def _find_env0_deformable_prim(self):
        for prim in self.scene.stage.Traverse():
            p = str(prim.GetPath())
            if "/env_0/" in p and getattr(self.cfg, "deformable_prim_token", "deformable_salmon") in p:
                a = prim.GetAttribute("physxDeformable:simulationPoints")
                if a and a.IsValid():
                    return prim
        return None

    def _build_deformable_surface_embedding(self, prim):
        """Barycentric embedding of the render-SURFACE rest points into the simulation tet
        mesh. Returns (node_indices (P,4) int32, weights (P,4) float32) or (None, None)."""
        sim_rest_attr = prim.GetAttribute("physxDeformable:simulationRestPoints")
        sim_idx_attr = prim.GetAttribute("physxDeformable:simulationIndices")
        surf_rest_attr = prim.GetAttribute("physxDeformable:restPoints")
        for a in (sim_rest_attr, sim_idx_attr, surf_rest_attr):
            if not a or not a.IsValid():
                return None, None
        sim_rest = sim_rest_attr.Get()
        sim_idx = sim_idx_attr.Get()
        surf_rest = surf_rest_attr.Get()
        if sim_rest is None or sim_idx is None or surf_rest is None:
            return None, None
        sim_rest = np.asarray(sim_rest, dtype=np.float64)
        tet_idx = np.asarray(sim_idx, dtype=np.int32).reshape(-1, 4)
        surf_rest = np.asarray(surf_rest, dtype=np.float64)
        if sim_rest.size == 0 or tet_idx.size == 0 or surf_rest.size == 0:
            return None, None
        node_indices, weights, inside, max_err = self._compute_tetrahedral_surface_embedding(
            surf_rest, sim_rest, tet_idx)
        print(f"[SalmonILWater] surface embedding: {surf_rest.shape[0]} surf pts into "
              f"{tet_idx.shape[0]} tets ({inside} inside, max reconstruction err {max_err:.4f} m)", flush=True)
        return node_indices, weights

    @staticmethod
    def _compute_tetrahedral_surface_embedding(surface_rest_points, simulation_rest_points, tet_indices, tol=1.0e-6):
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

    # --------------------------------------------- real / sim point cloud plumbing ----
    @staticmethod
    def _uniform_sample_indices(total_points: int, max_points: int) -> np.ndarray:
        if total_points <= 0:
            raise ValueError("Cannot sample from an empty point set.")
        if total_points <= max_points:
            return np.arange(total_points, dtype=np.int64)
        return np.linspace(0, total_points - 1, max_points, dtype=np.int64)

    def _load_real_pointcloud_sequence(self) -> None:
        pointcloud_dir = Path(self.cfg.real_pointcloud_dir)
        if not pointcloud_dir.exists():
            raise RuntimeError(f"Real-fish point cloud directory does not exist: {pointcloud_dir}")
        frame_paths = sorted(pointcloud_dir.glob(self.cfg.real_pointcloud_glob))
        stride = max(1, int(self.cfg.real_pointcloud_frame_stride))
        frame_paths = frame_paths[::stride]
        if not frame_paths:
            raise RuntimeError(f"No real-fish point clouds were found under {pointcloud_dir}")
        frames = []
        for fp in frame_paths:
            pts = np.load(fp)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise RuntimeError(f"Unexpected point cloud shape in {fp}: {pts.shape}")
            if not np.isfinite(pts).all():
                pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
            ids = self._uniform_sample_indices(pts.shape[0], self._il_point_count)
            frames.append(torch.as_tensor(pts[ids].astype(np.float32, copy=False),
                                          dtype=torch.float32, device=self.device))
        self._real_pointcloud_sequence = torch.stack(frames, dim=0)
        self._real_pointcloud_frame_count = self._real_pointcloud_sequence.shape[0]

    def _get_real_frame_indices(self) -> torch.Tensor:
        idx = self._il_real_frame_start + self._il_episode_frame
        if self.cfg.real_pointcloud_loop:
            return torch.remainder(idx, self._real_pointcloud_frame_count)
        return torch.clamp(idx, max=self._real_pointcloud_frame_count - 1)

    def _get_live_sim_pointclouds_for_reward(self) -> torch.Tensor:
        """Read the deformable simulation mesh nodal positions (world frame) via the
        inherited `_soft_view` and down-sample to the chamfer point set -> (E, P, 3)."""
        positions = self._soft_view.get_simulation_mesh_nodal_positions()
        if not isinstance(positions, torch.Tensor):
            positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        else:
            positions = positions.to(self.device)
        positions = positions[: self.num_envs]
        invalid = (~torch.isfinite(positions)).any(dim=(1, 2))
        self._reward_pointcloud_invalid |= invalid
        if not self._pointcloud_nan_reported and bool(invalid.any()):
            self._pointcloud_nan_reported = True
            print("[SalmonILWater] non-finite deformable reward vertices detected; clamping to zero.", flush=True)
        positions = torch.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        if self._sim_surface_node_indices is not None:
            # reconstruct the deformed SURFACE: (E, P, 4, 3) tet-node positions @ barycentric (P,4)
            embedded = positions[:, self._sim_surface_node_indices, :]
            pts = torch.einsum("pf,epfj->epj", self._sim_surface_weights, embedded)
            return torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        return positions[:, self._sim_reward_sample_indices, :]

    # ----------------------------------------- centerline / alignment / chamfer ----
    # (ported verbatim from salmon_IL.py -- physics-independent shape comparison)
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

    def _compute_alignment_transform(self, points: torch.Tensor, flip_major: bool = False):
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
        midpoint = torch.nan_to_num(midpoint, nan=0.0, posinf=0.0, neginf=0.0)
        rotation = torch.nan_to_num(rotation, nan=0.0, posinf=0.0, neginf=0.0)
        body_length = torch.nan_to_num(body_length, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1.0e-6)
        if not self._alignment_nan_reported and (
            (~torch.isfinite(midpoint)).any() or (~torch.isfinite(rotation)).any()
        ):
            print("[SalmonILWater] non-finite alignment transform detected; clamping.", flush=True)
            self._alignment_nan_reported = True
        return midpoint, rotation, body_length

    @staticmethod
    def _apply_alignment_transform(points, midpoint, rotation, scale):
        aligned = (torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0) - midpoint) @ rotation
        aligned = aligned / scale.clamp(min=1.0e-6)
        return torch.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _symmetric_chamfer_distance(points_a: torch.Tensor, points_b: torch.Tensor) -> torch.Tensor:
        points_a = torch.nan_to_num(points_a, nan=0.0, posinf=0.0, neginf=0.0)
        points_b = torch.nan_to_num(points_b, nan=0.0, posinf=0.0, neginf=0.0)
        pairwise = torch.cdist(points_a.unsqueeze(0), points_b.unsqueeze(0), p=2).squeeze(0)
        pairwise = torch.nan_to_num(pairwise, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)
        forward = pairwise.min(dim=1).values.mean()
        backward = pairwise.min(dim=0).values.mean()
        return torch.nan_to_num(0.5 * (forward + backward), nan=1.0e6, posinf=1.0e6, neginf=1.0e6)

    @staticmethod
    def _sign_candidates(rot: torch.Tensor):
        """The 4 right-handed rotations over the (+/-major, +/-secondary) sign ambiguity of an
        SVD/centerline frame (rot columns = [major, secondary, third]). Order: (k>=2) == major flipped."""
        major0, sec0 = rot[:, 0], rot[:, 1]
        out = []
        for sm in (1.0, -1.0):
            for ss in (1.0, -1.0):
                major = major0 * sm
                sec = sec0 * ss
                third = torch.cross(major, sec, dim=0)
                out.append(torch.stack((major, sec, third), dim=1))
        return out

    def _pick_consistent_rotation(self, rot0, prev_rot, prev_valid):
        """Resolve the SVD sign ambiguity by choosing the sign combo whose rotation is closest to
        the previous frame's (max trace(prev^T R)) -> no head-tail / up-down flips between frames."""
        cands = self._sign_candidates(rot0)
        if not bool(prev_valid.item()):
            return cands[0]
        best, best_score = cands[0], float("-inf")
        for R in cands:
            score = float((prev_rot * R).sum())   # == trace(prev_rot^T @ R)
            if score > best_score:
                best, best_score = R, score
        return best

    # ===================== VECTORISED alignment + chamfer (all envs at once) =========
    # Validated to match the per-env reference to ~1e-7; ~16-20x faster (the per-env python
    # loop with .item() syncs took ~3.4s/step at 64 envs -> the "stuck in one epoch").
    def _centerline_batch(self, pts):
        """pts (E,N,3) -> (midpoint (E,3), major (E,3 unit), Vh (E,3,3), scale (E,) arc-length)."""
        E, N, _ = pts.shape
        nb = int(self.cfg.centerline_num_bins)
        minp = max(1, int(self.cfg.centerline_min_bin_points))
        pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        centroid = pts.mean(1)
        centered = pts - centroid[:, None]
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)         # (E,3,3)
        except RuntimeError:
            Vh = torch.eye(3, device=pts.device)[None].expand(E, -1, -1).contiguous()
        major0 = Vh[:, 0]
        major0 = major0 / major0.norm(dim=1, keepdim=True).clamp(min=1e-6)
        proj = (centered * major0[:, None]).sum(-1)                            # (E,N)
        pmn = proj.min(1, keepdim=True).values
        pmx = proj.max(1, keepdim=True).values
        rng = (pmx - pmn).clamp(min=1e-6)
        bidx = (((proj - pmn) / rng) * nb).long().clamp(0, nb - 1)             # (E,N)
        binsum = torch.zeros(E, nb, 3, device=pts.device).scatter_add_(1, bidx[..., None].expand(-1, -1, 3), pts)
        cnt = torch.zeros(E, nb, device=pts.device).scatter_add_(1, bidx, torch.ones_like(proj))
        valid = cnt >= minp
        binmean = binsum / cnt.clamp(min=1)[..., None]
        ar = torch.arange(nb, device=pts.device)[None].expand(E, -1)
        first = torch.where(valid, ar, torch.full_like(ar, nb + 1)).min(1).values.clamp(max=nb - 1)
        last = torch.where(valid, ar, torch.full_like(ar, -1)).max(1).values.clamp(min=0)
        bi = torch.arange(E, device=pts.device)
        pf = binmean[bi, first]
        pl = binmean[bi, last]
        mid = 0.5 * (pf + pl)
        major = pl - pf
        deg = major.norm(dim=1) < 1e-6
        major = torch.where(deg[:, None], major0, major)
        major = major / major.norm(dim=1, keepdim=True).clamp(min=1e-6)
        # arc length over CONSECUTIVE VALID bins (connect across gaps): forward-fill invalid bins
        # with the last valid bin (cummax of valid index), then sum consecutive diffs.
        vidx = torch.where(valid, ar, torch.full_like(ar, -1))
        lastv = torch.cummax(vidx, dim=1).values
        lastv = torch.where(lastv < 0, first[:, None], lastv)
        filled = torch.gather(binmean, 1, lastv[..., None].expand(-1, -1, 3))
        diffs = filled[:, 1:] - filled[:, :-1]
        arc = diffs.norm(dim=2).sum(1)
        chord = (pl - pf).norm(dim=1)
        scale = torch.where(arc > 1e-6, arc, chord).clamp(min=1e-6)
        return torch.nan_to_num(mid), major, Vh, scale

    @staticmethod
    def _build_rot_batch(major, sec0):
        sec = sec0 - (sec0 * major).sum(1, keepdim=True) * major
        sec = sec / sec.norm(dim=1, keepdim=True).clamp(min=1e-6)
        third = torch.cross(major, sec, dim=1)
        third = third / third.norm(dim=1, keepdim=True).clamp(min=1e-6)
        sec = torch.cross(third, major, dim=1)
        sec = sec / sec.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return torch.stack((major, sec, third), dim=2)                        # (E,3,3) columns

    @staticmethod
    def _sign_candidates_batch(rot):
        major, sec = rot[:, :, 0], rot[:, :, 1]
        out = []
        for sm in (1.0, -1.0):
            for ss in (1.0, -1.0):
                m = major * sm
                s = sec * ss
                t = torch.cross(m, s, dim=1)
                out.append(torch.stack((m, s, t), dim=2))
        return torch.stack(out, dim=1)                                        # (E,4,3,3)

    @staticmethod
    def _apply_batch(pts, mid, rot, scale):
        pts = torch.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(torch.einsum('epj,ejk->epk', pts - mid[:, None], rot) / scale[:, None, None].clamp(min=1e-6))

    @staticmethod
    def _chamfer_batch(a, b):
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        pw = torch.nan_to_num(torch.cdist(a, b), nan=1e6, posinf=1e6, neginf=1e6)   # (E,P,P)
        return 0.5 * (pw.min(2).values.mean(1) + pw.min(1).values.mean(1))          # (E,)

    def _align_and_chamfer_batch(self, real_pts, sim_pts):
        """All envs at once. Real signs kept temporally consistent (no flips); sim matched to real
        by the min-chamfer of the 4 sign flips. Returns (real_aligned, sim_aligned, chamfer, flip)."""
        E = real_pts.shape[0]
        bi = torch.arange(E, device=real_pts.device)
        r_mid, r_maj, r_Vh, r_scale = self._centerline_batch(real_pts)
        r_cands = self._sign_candidates_batch(self._build_rot_batch(r_maj, r_Vh[:, 1]))   # (E,4,3,3)
        scores = (self._prev_real_rot[:, None] * r_cands).sum((2, 3))                      # (E,4) trace(prev^T R)
        best_r = torch.where(self._prev_real_valid, scores.argmax(1), torch.zeros(E, dtype=torch.long, device=real_pts.device))
        r_rot = r_cands[bi, best_r]
        self._prev_real_rot = r_rot.detach()
        self._prev_real_valid[:] = True
        real_aligned = self._apply_batch(real_pts, r_mid, r_rot, r_scale)

        s_mid, s_maj, s_Vh, s_scale = self._centerline_batch(sim_pts)
        s_cands = self._sign_candidates_batch(self._build_rot_batch(s_maj, s_Vh[:, 1]))   # (E,4,3,3)
        chs, sas = [], []
        for k in range(4):
            sa = self._apply_batch(sim_pts, s_mid, s_cands[:, k], s_scale)
            sas.append(sa)
            chs.append(self._chamfer_batch(real_aligned, sa))
        chs = torch.stack(chs, 1)                                                          # (E,4)
        best_s = chs.argmin(1)
        sim_aligned = torch.stack(sas, 1)[bi, best_s]
        return real_aligned, sim_aligned, chs[bi, best_s], (best_s >= 2)

    def _align_pair_and_chamfer(self, env_id, real_points, sim_points):
        """Re-canonicalise real & sim THIS frame (centerline center/rotate/scale) and return
        (real_aligned, sim_aligned, chamfer, flip). The real frame's axis signs are kept TEMPORALLY
        CONSISTENT (no frame-to-frame head-tail / up-down flips); the sim is matched to the real by
        searching all 4 sign flips and picking the min-chamfer one (so sim is never mirrored vs real).
        Recomputed every step so the comparison is pure body-bend shape, invariant to global pose."""
        r_mid, r_rot0, r_scale = self._compute_alignment_transform(real_points)
        real_rot = self._pick_consistent_rotation(r_rot0, self._prev_real_rot[env_id], self._prev_real_valid[env_id])
        self._prev_real_rot[env_id] = real_rot
        self._prev_real_valid[env_id] = True
        real_aligned = self._apply_alignment_transform(real_points, r_mid, real_rot, r_scale)

        s_mid, s_rot0, s_scale = self._compute_alignment_transform(sim_points)
        best_ch = best_sim = None
        best_flip = False
        for k, R in enumerate(self._sign_candidates(s_rot0)):
            sa = self._apply_alignment_transform(sim_points, s_mid, R, s_scale)
            ch = self._symmetric_chamfer_distance(real_aligned, sa)
            if best_ch is None or bool((ch < best_ch).item()):
                best_ch, best_sim, best_flip = ch, sa, (k >= 2)
        return real_aligned, best_sim, best_ch, best_flip

    def _get_reward_aligned_pointclouds(self, env_id, real_points, sim_points):
        real_aligned = self._apply_alignment_transform(
            real_points, self._real_alignment_midpoint[env_id],
            self._real_alignment_rotation[env_id], self._real_alignment_scale[env_id])
        sim_aligned = self._apply_alignment_transform(
            sim_points, self._sim_alignment_midpoint[env_id],
            self._sim_alignment_rotation[env_id], self._sim_alignment_scale[env_id])
        return real_aligned, sim_aligned

    def _initialize_episode_alignment(self, env_id, real_points, sim_points):
        real_mid, real_rot, real_scale = self._compute_alignment_transform(real_points, flip_major=False)
        real_aligned = self._apply_alignment_transform(real_points, real_mid, real_rot, real_scale)
        sim_mid_d, sim_rot_d, sim_scale_d = self._compute_alignment_transform(sim_points, flip_major=False)
        sim_aligned_d = self._apply_alignment_transform(sim_points, sim_mid_d, sim_rot_d, sim_scale_d)
        chamfer_d = self._symmetric_chamfer_distance(real_aligned, sim_aligned_d)
        sim_mid_f, sim_rot_f, sim_scale_f = self._compute_alignment_transform(sim_points, flip_major=True)
        sim_aligned_f = self._apply_alignment_transform(sim_points, sim_mid_f, sim_rot_f, sim_scale_f)
        chamfer_f = self._symmetric_chamfer_distance(real_aligned, sim_aligned_f)
        use_flipped = bool((chamfer_f < chamfer_d).item())
        if use_flipped:
            sim_mid, sim_rot, sim_scale, sim_aligned, baseline = sim_mid_f, sim_rot_f, sim_scale_f, sim_aligned_f, chamfer_f
        else:
            sim_mid, sim_rot, sim_scale, sim_aligned, baseline = sim_mid_d, sim_rot_d, sim_scale_d, sim_aligned_d, chamfer_d
        self._real_alignment_midpoint[env_id] = real_mid
        self._real_alignment_rotation[env_id] = real_rot
        self._real_alignment_scale[env_id] = real_scale
        self._sim_alignment_midpoint[env_id] = sim_mid
        self._sim_alignment_rotation[env_id] = sim_rot
        self._sim_alignment_scale[env_id] = sim_scale
        self._sim_alignment_flip_major[env_id] = use_flipped
        self._alignment_initialized[env_id] = True
        # NOTE: the reward baseline is the MEAN of the first `baseline_num_frames` post-warmup
        # chamfers, accumulated in _get_rewards -- NOT this single-frame value.
        return real_aligned, sim_aligned, baseline

    # ------------------------------------------------------ overrides: obs/reward ----
    def _configure_gym_env_spaces(self):
        self._control_joint_ids = torch.arange(self.robot.num_joints, device=self.device)
        self._num_actions = self._control_joint_ids.shape[0]
        # proprio: grav(3)+lin(3)+ang(3)+joint_pos_err(nj)+joint_vel(nj)  +  phase(2)  [+ target(6)]
        self._use_target_obs = bool(getattr(self.cfg, "demo_path", "")) and bool(getattr(self.cfg, "demo_target_obs", False))
        obs_dim = 11 + 2 * self._num_actions + (6 if self._use_target_obs else 0)
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(obs_dim,))}
        )
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space["policy"], self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actions,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_actions), device=self.device)
        self.state_space = None

    def _get_observations(self) -> dict:
        root_state = self.robot.data.root_state_w
        projected_gravity = self.robot.data.projected_gravity_b
        root_lin_vel = root_state[:, 7:10] * self.cfg.obs_scales.root_lin_vel
        root_ang_vel = root_state[:, 10:13] * self.cfg.obs_scales.root_ang_vel
        joint_pos_error = (self.joint_pos - self._default_joint_pos) * self.cfg.obs_scales.joint_pos
        joint_vel = self.joint_vel * self.cfg.obs_scales.joint_vel
        # imitation phase: progress through the demo (demo mode) or the looped real clip
        if getattr(self, "_demo_mode", False):
            frac = (self._demo_start_frame.float() + self.episode_length_buf.float() - 1.0).clamp(min=0.0) / max(1, self._demo_T)
        else:
            frac = self._get_real_frame_indices().float() / max(1, self._real_pointcloud_frame_count)
        angle = 2.0 * torch.pi * frac
        phase = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        parts = [projected_gravity, root_lin_vel, root_ang_vel, joint_pos_error, joint_vel, phase]
        # Egocentric MOVING-REFERENCE target (+6): body-frame vector to the demo's CURRENT position
        # + the demo's heading in the body frame. Makes the relative offset the reward penalizes
        # OBSERVABLE (translation-invariant -> deployable). See cfg.demo_target_obs.
        if getattr(self, "_use_target_obs", False):
            if getattr(self, "_demo_mode", False) and getattr(self, "_demo_has_full", False):
                t = (self._demo_start_frame + self.episode_length_buf - 1).clamp(min=0, max=self._demo_T - 1)  # (E,)
                root_pos_local = root_state[:, 0:3] - self.scene.env_origins      # env-local (demo is env0-local)
                root_quat = root_state[:, 3:7]                                    # (E,4) wxyz
                demo = self._demo_root_t[t]                                       # (E,13) env-local
                pos_err_world = demo[:, 0:3] - root_pos_local                     # (E,3) demo - sim  (a DIFFERENCE)
                pos_err_body = math_utils.quat_apply_inverse(root_quat, pos_err_world)   # egocentric
                fwd = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3)
                demo_fwd_world = math_utils.quat_apply(demo[:, 3:7], fwd)         # demo heading (world)
                demo_fwd_body = math_utils.quat_apply_inverse(root_quat, demo_fwd_world)  # demo heading in MY frame
                parts.append(pos_err_body); parts.append(demo_fwd_body)
            else:
                parts.append(torch.zeros(self.num_envs, 6, device=self.device))
        obs = torch.cat(parts, dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return {"policy": obs}

    def _demo_rewards(self) -> torch.Tensor:
        """DEMO mode: reward = exp(-world_chamfer / temp) between the IL fish's FEM cloud and the
        RL-demo cloud at the same timestep -- WORLD frame, NO alignment (position+pose must match)."""
        sim_pc = self._get_live_sim_pointclouds_for_reward()                     # (E,N,3) world
        # to ENV-LOCAL frame (remove the parallel-env tiling offset; demo was recorded at env0,
        # origin 0). This is NOT pose alignment -- the fish's position WITHIN its env is preserved.
        sim_pc = sim_pc - self.scene.env_origins[:, None, :]
        # demo frame = RSI start offset + steps since reset (each env follows the demo from its k)
        t = (self._demo_start_frame + self.episode_length_buf - 1).clamp(min=0, max=self._demo_T - 1)  # (E,)
        demo_pc = self._demo_pc[t]                                               # (E,N,3) world
        chamfer = torch.nan_to_num(self._chamfer_batch(sim_pc, demo_pc), nan=1e6, posinf=1e6, neginf=1e6)
        # far-field: world distance between the two clouds' centroids -> always-on gradient even
        # when there is zero overlap (exp(-chamfer/temp) is flat there).
        d_centroid = torch.nan_to_num((sim_pc.mean(1) - demo_pc.mean(1)).norm(dim=1), nan=1e6, posinf=1e6, neginf=1e6)
        self._demo_last_centroid = d_centroid.detach()                          # for Early Termination in _get_dones

        # ---- velocity (linear) + head->tail axis (angular) terms: per-step ACTION-coupled ----
        self._ensure_headtail()
        head = self._head_mask
        sim_cen = sim_pc.mean(1)                                                # (E,3)
        sim_ax = sim_pc[:, head].mean(1) - sim_pc[:, ~head].mean(1)
        sim_ax = sim_ax / sim_ax.norm(dim=1, keepdim=True).clamp(min=1e-6)      # (E,3) head<-tail unit axis
        sim_vel = (sim_cen - self._prev_sim_centroid) / self._dt_ctrl          # (E,3)
        sim_avel = (sim_ax - self._prev_sim_axis) / self._dt_ctrl
        demo_vel = self._demo_cvel[t]                                           # (E,3)
        demo_avel = self._demo_avel[t]
        vel_err = (sim_vel - demo_vel).norm(dim=1)
        avel_err = (sim_avel - demo_avel).norm(dim=1)
        tv = max(1e-6, float(self.cfg.demo_vel_temp)); ta = max(1e-6, float(self.cfg.demo_avel_temp))
        r_vel = torch.where(self._prev_vel_valid, torch.exp(-(vel_err ** 2) / tv), torch.zeros_like(vel_err))
        r_avel = torch.where(self._prev_vel_valid, torch.exp(-(avel_err ** 2) / ta), torch.zeros_like(avel_err))

        temp = max(1.0e-6, float(self.cfg.demo_reward_temp))
        w_far = float(self.cfg.demo_far_weight)
        if str(getattr(self.cfg, "demo_reward_kind", "rational")) == "exp":
            near = torch.exp(-chamfer / temp)
        else:                                               # rational: 1/(1+chamfer/temp), heavy tail
            near = 1.0 / (1.0 + chamfer / temp)
        reward = (near - w_far * d_centroid
                  + float(self.cfg.demo_vel_weight) * r_vel
                  + float(self.cfg.demo_avel_weight) * r_avel)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        self._prev_sim_centroid = sim_cen.detach()
        self._prev_sim_axis = sim_ax.detach()
        self._prev_vel_valid[:] = True

        self._episode_rewards += reward
        self._il_step_counter = getattr(self, "_il_step_counter", 0) + 1
        chamfer_mean = chamfer.mean()
        self.extras["log"] = {"demo_chamfer_mean": chamfer_mean, "demo_centroid_dist_mean": d_centroid.mean(),
                              "demo_vel_err_mean": vel_err.mean(), "demo_avel_err_mean": avel_err.mean(),
                              "reward_total_mean": reward.mean()}
        if self.cfg.debug_logs and (self._il_step_counter % 10 == 0):
            jvmax = float(self.joint_vel.abs().max())
            epoch = self._il_step_counter // max(1, int(self.cfg.video_clip_steps))   # ~rl_games epoch
            print(f"[SalmonILWater-DEMO] ep{epoch} step {self._il_step_counter}: chamfer={float(chamfer_mean):.4f} "
                  f"centroid_d={float(d_centroid.mean()):.4f} vel_err={float(vel_err.mean()):.4f} "
                  f"avel_err={float(avel_err.mean()):.4f} reward={float(reward.mean()):.4f} "
                  f"t0={int(t[0])}/{self._demo_T} jvel_max={jvmax:.2f} "
                  f"cum_blowups={getattr(self,'_blowup_total',0)} oob={getattr(self,'_oob_total',0)}", flush=True)
        if getattr(self, "_monitor", False):
            vec0 = {
                "sim_cen": sim_cen[0].detach().cpu().numpy(), "demo_cen": demo_pc[0].mean(0).detach().cpu().numpy(),
                "sim_vel": sim_vel[0].detach().cpu().numpy(), "demo_vel": demo_vel[0].detach().cpu().numpy(),
                "sim_axis": sim_ax[0].detach().cpu().numpy(), "demo_axis": self._demo_axis_seq[t[0]].detach().cpu().numpy(),
            }
            self._monitor_step(demo_pc[0], sim_pc[0], int(t[0].item()), False, float(chamfer[0].item()), vec0)
        return reward

    def _ensure_headtail(self) -> None:
        """Label the FEM nodes head/tail ONCE, using the skeleton bone order to set the axis sign
        (anatomically consistent; real clouds are assumed pre-labelled). Precompute the demo's
        centroid-velocity and head->tail-axis-velocity sequences."""
        if self._head_mask is not None:
            return
        bp = self.robot.data.body_pos_w[0]                                     # (B,3) world bone positions
        axis = bp[0] - bp[-1]                                                  # head (root, body 0) -> tail
        axis = axis / axis.norm().clamp(min=1e-6)
        ref = self._demo_pc[0]                                                 # (N,3) env-local rest-ish cloud
        proj = (ref - ref.mean(0)) @ axis
        self._head_mask = proj > proj.median()                                # (N,) bool: head half
        head = self._head_mask
        cen = self._demo_pc.mean(1)                                            # (T,3)
        ax = self._demo_pc[:, head].mean(1) - self._demo_pc[:, ~head].mean(1)
        ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-6)                 # (T,3) head<-tail unit
        cvel = torch.zeros_like(cen); cvel[1:] = (cen[1:] - cen[:-1]) / self._dt_ctrl
        avel = torch.zeros_like(ax);  avel[1:] = (ax[1:] - ax[:-1]) / self._dt_ctrl
        self._demo_axis_seq, self._demo_cvel, self._demo_avel = ax, cvel, avel
        print(f"[SalmonILWater] head/tail labelled from skeleton: {int(head.sum())} head / {int((~head).sum())} tail "
              f"nodes | demo |vel|~{float(cvel.norm(dim=1).mean()):.3f} m/s, |avel|~{float(avel.norm(dim=1).mean()):.3f}/s",
              flush=True)

    def _get_rewards(self) -> torch.Tensor:
        if getattr(self, "_demo_mode", False):
            return self._demo_rewards()
        sim_pointclouds = self._get_live_sim_pointclouds_for_reward()
        # DIAG: is the raw sim cloud actually changing step-to-step?
        if self.cfg.debug_logs:
            raw0 = sim_pointclouds[0]
            sraw0 = raw0 - raw0.mean(0, keepdim=True)   # centroid-removed -> SHAPE only
            self._dbg_d_raw = (float((raw0 - self._dbg_prev_raw0).abs().mean())
                               if hasattr(self, "_dbg_prev_raw0") else float("nan"))
            self._dbg_d_shape = (float((sraw0 - self._dbg_prev_sraw0).abs().mean())
                                 if hasattr(self, "_dbg_prev_sraw0") else float("nan"))
            self._dbg_prev_raw0 = raw0.clone()
            self._dbg_prev_sraw0 = sraw0.clone()
        real_frame_indices = self._get_real_frame_indices()
        warmup_mask = self._il_env_step < self._il_warmup_steps
        # PER-FRAME, ALL-ENV alignment + chamfer (vectorised): re-canonicalise both clouds every
        # step so chamfer measures body-bend shape (robust to the fish reorienting), real signs kept
        # temporally consistent, sim matched to real over the 4 sign flips.
        real_pts = self._real_pointcloud_sequence[real_frame_indices]            # (E,P,3)
        real_aligned, sim_aligned, current_chamfer, flip = self._align_and_chamfer_batch(real_pts, sim_pointclouds)
        current_chamfer = torch.where(warmup_mask, torch.zeros_like(current_chamfer), current_chamfer)
        self._sim_alignment_flip_major[0] = flip[0]
        ov_real0, ov_sim0 = real_aligned[0], sim_aligned[0]   # env0 clouds for the overlay monitor

        if not self._reward_nan_reported and bool((~torch.isfinite(current_chamfer)).any()):
            self._reward_nan_reported = True
            print(f"[SalmonILWater] non-finite Chamfer at step {self._il_step_counter}; clamping.", flush=True)
        current_chamfer = torch.nan_to_num(current_chamfer, nan=1.0e6, posinf=1.0e6, neginf=1.0e6)

        # PER-FRAME baseline-improvement reward. current_chamfer is always sim-vs-CURRENT-real-frame
        # (per-frame tracking -> dense target). The baseline is the MEAN of the first
        # `baseline_num_frames` post-warmup chamfers of this episode (de-noises the single-frame
        # baseline). During that accumulation window reward = 0; once frozen, pay only the
        # improvement below the baseline each frame, so the start earns 0 and tracking the gait
        # better than the (averaged) start is rewarded.
        post = ~warmup_mask
        accumulating = post & (~self._baseline_ready)
        self._baseline_sum[accumulating] += current_chamfer[accumulating]
        self._baseline_n[accumulating] += 1
        just_ready = accumulating & (self._baseline_n >= self._baseline_num_frames)
        self._chamfer_baseline[just_ready] = self._baseline_sum[just_ready] / self._baseline_n[just_ready].float()
        self._baseline_ready[just_ready] = True
        self._chamfer_baseline_valid[just_ready] = True

        base = self._chamfer_baseline
        if self.cfg.chamfer_reward_relative:
            improvement = ((base - current_chamfer) / base.clamp(min=1.0e-3)).clamp(0.0, 1.0)
        else:
            improvement = (base - current_chamfer).clamp(min=0.0)
        reward = self._chamfer_reward_weight * improvement
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        # zero reward until the baseline is frozen (covers warmup AND the accumulation window)
        reward[~self._baseline_ready] = 0.0

        # advance per-env IL pointers (frame only after warmup)
        self._current_chamfer = current_chamfer
        post = ~warmup_mask
        self._best_chamfer[post] = torch.minimum(self._best_chamfer[post], current_chamfer[post])
        self._episode_rewards += reward
        self._il_episode_frame += post.to(dtype=self._il_episode_frame.dtype)
        self._il_env_step += 1
        self._il_step_counter += 1

        chamfer_mean = torch.nan_to_num(current_chamfer[post].mean() if bool(post.any()) else current_chamfer.mean(),
                                        nan=1.0e6, posinf=1.0e6, neginf=0.0)
        reward_mean = torch.nan_to_num(reward.mean(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg.debug_logs and (self._il_step_counter % 10 == 0):
            jvmax = float(self.joint_vel.abs().max())
            actmax = float(self.actions.abs().max())
            actmean = float(self.actions.abs().mean())
            sig = float(getattr(self, "_dbg_sigma", float("nan")))
            rootspd = float(self.robot.data.root_state_w[:, 7:10].norm(dim=1).mean())
            base_mean = float(self._chamfer_baseline[~warmup_mask].mean()) if bool((~warmup_mask).any()) else 0.0
            d_raw = float(getattr(self, "_dbg_d_raw", float("nan")))
            d_shape = float(getattr(self, "_dbg_d_shape", float("nan")))
            print(f"[SalmonILWater] step {self._il_step_counter}: chamfer={float(chamfer_mean):.4f} "
                  f"baseline={base_mean:.4f} reward={float(reward_mean):.4f} | jvel_max={jvmax:.3f} "
                  f"root_spd={rootspd:.4f} m/s sim_dstep={d_raw:.6f} sim_SHAPE_dstep={d_shape:.6f} "
                  f"warmup_envs={int(warmup_mask.sum())}", flush=True)

        self.extras["log"] = {
            "chamfer_distance_mean": chamfer_mean,
            "chamfer_best_mean": torch.nan_to_num(self._best_chamfer.mean(), nan=1.0e6, posinf=1.0e6, neginf=0.0),
            "reward_total_mean": reward_mean,
        }

        if getattr(self, "_monitor", False):
            self._monitor_step(ov_real0, ov_sim0, int(real_frame_indices[0].item()),
                               bool(warmup_mask[0].item()), float(current_chamfer[0].item()))
        return reward

    # ----------------------------------------------------- overrides: dones/reset ----
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        root_state = self.robot.data.root_state_w
        if getattr(self, "_demo_mode", False):
            # RSI: each env follows the demo from its start frame -> ends when it reaches the last frame
            time_out = (self._demo_start_frame + self.episode_length_buf) >= self._demo_T
        else:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        local_pos = root_state[:, 0:3] - self.scene.env_origins
        out_of_bounds = torch.linalg.norm(local_pos, dim=1) > self.cfg.root_position_limit
        joint_blowup = self.joint_vel.abs().max(dim=1).values > self.cfg.joint_velocity_limit
        nonfinite = (~torch.isfinite(root_state).all(dim=1)) | (~torch.isfinite(self.joint_pos).all(dim=1)) \
            | (~torch.isfinite(self.joint_vel).all(dim=1))
        terminated = out_of_bounds | joint_blowup | nonfinite | self._reward_pointcloud_invalid
        # ---- blow-up accounting (IL env previously logged NOTHING on blow-up) ----
        _blew = joint_blowup | nonfinite
        self._blew_now = int(_blew.sum())
        self._blowup_total = getattr(self, "_blowup_total", 0) + self._blew_now
        self._oob_total = getattr(self, "_oob_total", 0) + int(out_of_bounds.sum())
        if self._blew_now:   # ALWAYS print immediately (every step), so a spike can't be missed
            jvm = float(self.joint_vel.abs().max())
            njb = int(joint_blowup.sum()); nnf = int(nonfinite.sum())
            _ep = getattr(self, "_il_step_counter", 0) // max(1, int(self.cfg.video_clip_steps))
            print(f"[SalmonILWater] <<< BLOWUP ep{_ep} step {getattr(self,'_il_step_counter',0)}: "
                  f"{self._blew_now} env(s) (joint_vel>200: {njb}, nonfinite: {nnf}), jvel_max={jvm:.1f}, "
                  f"cum_blowups={self._blowup_total}", flush=True)
        # Early Termination: drifted too far from the demo's current position
        if getattr(self, "_demo_mode", False) and float(self.cfg.demo_et_centroid) > 0:
            terminated = terminated | (self._demo_last_centroid > float(self.cfg.demo_et_centroid))
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # base reset: robot pose/joints, soft-body teleport, swim episode stats
        super()._reset_idx(env_ids)
        if not hasattr(self, "_il_env_step"):
            return  # IL state not built yet (first call is post-__init__, so normally set)
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        self._il_env_step[env_ids] = 0
        self._il_episode_frame[env_ids] = 0
        self._alignment_initialized[env_ids] = False
        self._prev_real_valid[env_ids] = False
        self._chamfer_baseline_valid[env_ids] = False
        self._chamfer_baseline[env_ids] = 0.0
        self._baseline_sum[env_ids] = 0.0
        self._baseline_n[env_ids] = 0
        self._baseline_ready[env_ids] = False
        self._best_chamfer[env_ids] = float("inf")
        self._reward_pointcloud_invalid[env_ids] = False
        eye = torch.eye(3, device=self.device)
        self._real_alignment_rotation[env_ids] = eye
        self._sim_alignment_rotation[env_ids] = eye
        self._real_alignment_midpoint[env_ids] = 0.0
        self._sim_alignment_midpoint[env_ids] = 0.0
        self._real_alignment_scale[env_ids] = 1.0
        self._sim_alignment_scale[env_ids] = 1.0
        if self.cfg.real_pointcloud_random_start:
            self._il_real_frame_start[env_ids] = torch.randint(
                0, self._real_pointcloud_frame_count, (len(env_ids),), device=self.device, dtype=torch.long)
        else:
            self._il_real_frame_start[env_ids] = int(self.cfg.real_pointcloud_start_frame)

        # DEMO-RSI: place reset envs at a RANDOM demo frame's full state (frame 0 if RSI off).
        if getattr(self, "_demo_mode", False):
            m = len(env_ids)
            if getattr(self, "_demo_rsi", False):
                k = torch.randint(0, self._demo_T, (m,), device=self.device, dtype=torch.long)
            else:
                k = torch.zeros(m, dtype=torch.long, device=self.device)
            self._demo_start_frame[env_ids] = k
            self._demo_last_centroid[env_ids] = 0.0
            self._prev_vel_valid[env_ids] = False        # first post-reset step has no prev velocity
            if getattr(self, "_demo_has_full", False):
                self._set_demo_state(env_ids, k)

        # env0 trajectory snapshot at episode end (kept; previous saves are not deleted)
        if getattr(self, "_monitor", False):
            ids = env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
            if 0 in ids:
                self._save_traj_snapshot()
                self._traj_xyz = []

    def _set_demo_state(self, env_ids, k):
        """RSI: set the given envs to demo frame k's FULL state (root, joints, FEM nodal pos+vel).
        Demo is recorded at env0 (origin 0); shift positions by each env's origin."""
        root = self._demo_root_t[k].clone()                                   # (m,13) env0-local
        root[:, 0:3] = root[:, 0:3] + self.scene.env_origins[env_ids]         # -> world per env
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:13], env_ids)
        jpos = self._demo_jpos[k]
        jvel = self._demo_jvel[k]
        self.joint_pos[env_ids] = jpos
        self.joint_vel[env_ids] = jvel
        self.robot.write_joint_state_to_sim(jpos, jvel, None, env_ids)
        if getattr(self, "_soft_view", None) is not None:
            try:
                pos = self._soft_view.get_simulation_mesh_nodal_positions()
                vel = self._soft_view.get_simulation_mesh_nodal_velocities()
                nodal = self._demo_pc[k] + self.scene.env_origins[env_ids][:, None, :]   # (m,N,3) world
                pos[env_ids] = nodal.to(pos.dtype)
                vel[env_ids] = self._demo_nvel[k].to(vel.dtype)
                self._soft_view.set_simulation_mesh_nodal_positions(pos)
                self._soft_view.set_simulation_mesh_nodal_velocities(vel)
            except Exception as e:  # noqa: BLE001
                print(f"[SalmonILWater] RSI FEM state-set failed: {e}", flush=True)

    # ============================================================ monitoring ========
    def _monitor_setup(self) -> None:
        self._monitor = bool(getattr(self.cfg, "monitor_enabled", False))
        if not self._monitor:
            return
        import os, time as _t
        self._run_dir = os.path.join(self.cfg.outcome_dir, _t.strftime("run_%Y%m%d_%H%M%S"))
        self._traj_dir = os.path.join(self._run_dir, "trajectory_env0")
        os.makedirs(self._traj_dir, exist_ok=True)
        self._overlay_dir = self._run_dir
        os.makedirs(self._overlay_dir, exist_ok=True)
        # convenience: outcome/latest -> this run, so you never watch a stale folder from a dead run
        try:
            link = os.path.join(self.cfg.outcome_dir, "latest")
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.basename(self._run_dir), link)
        except OSError:
            pass
        self._traj_xyz = []          # env0 env-local root path for the current episode
        self._traj_ep = 0
        self._ov_buf = []            # buffered env0 (real_aligned, sim_aligned) records
        self._ov_buf_epoch = 0
        self._ov_frame = 0           # monotone count of env0 post-warmup frames recorded
        proj = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
        self._ov_axes = proj.get(str(self.cfg.overlay_projection).lower(), (0, 1))
        print(f"[SalmonILWater] monitor -> {self._run_dir} | traj + overlay saved per episode "
              f"(episodes 0,1,2,5,10,15,... ; overlay <= {self.cfg.overlay_window} cells, "
              f"{self.cfg.overlay_projection}, with velocity+orientation arrows)", flush=True)

    def _monitor_step(self, real0, sim0, real_frame0: int, warmup0: bool, chamfer0: float, vec0=None) -> None:
        # ---- env0 trajectory ----
        root0 = (self.robot.data.root_state_w[0, 0:3] - self.scene.env_origins[0]).detach().cpu().numpy()
        self._traj_xyz.append(root0.copy())
        if (self._il_step_counter % max(1, int(self.cfg.traj_save_every))) == 0:
            self._save_traj_live()
        # ---- env0 reward overlay (accumulated over the episode; flushed at episode end) ----
        if (not warmup0) and (real0 is not None) and (sim0 is not None):
            self._record_overlay(real0, sim0, real_frame0, chamfer0, vec0)

    # ---- env0 trajectory plotting (PNG only; no OBJ) ----
    def _save_traj_live(self) -> None:
        self._plot_traj_il("traj_live")

    @staticmethod
    def _traj_ep_saved(ep: int) -> bool:
        # first 3 episodes, then every 5th: 0,1,2,5,10,15,...
        return ep < 3 or ep % 5 == 0

    def _save_traj_snapshot(self) -> None:
        ep = self._traj_ep
        self._traj_ep += 1
        saved = self._traj_ep_saved(ep)
        if saved and len(self._traj_xyz) >= 2:
            self._plot_traj_il(f"traj_ep{ep:04d}")                 # IL own trajectory
            if getattr(self, "_demo_mode", False) and getattr(self, "_demo_root_np", None) is not None:
                self._plot_traj_compare(f"traj_cmp_ep{ep:04d}")    # IL vs GT demo
        # overlay grid for THIS episode, on the SAME schedule as the trajectory plots
        if saved and len(self._ov_buf) >= 1:
            try:
                self._write_overlay_grid(ep)
            except Exception as e:  # noqa: BLE001
                print(f"[SalmonILWater] overlay grid save failed: {e}", flush=True)
        self._ov_buf = []   # fresh buffer for the next episode (cleared every episode)

    def _plot_traj_il(self, stem: str) -> None:
        import os
        if len(self._traj_xyz) < 2:
            return
        traj = np.asarray(self._traj_xyz, dtype=np.float32)
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-", color="tab:orange", lw=1.5, label=f"IL path [{len(traj)}]")
            ax.scatter(*traj[0], c="green", s=70, label="start", depthshade=False)
            ax.scatter(*traj[-1], c="red", s=45, label="end", depthshade=False)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
            ax.set_title(f"env0 IL path  ({len(traj)} steps)")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(self._traj_dir, stem + ".png"), dpi=120); plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"[SalmonILWater] traj PNG save failed: {e}", flush=True)

    def _plot_traj_compare(self, stem: str) -> None:
        """IL fish vs GT demo trajectory in one figure (3D + top-down xy)."""
        import os
        il = np.asarray(self._traj_xyz, dtype=np.float32)
        gt = self._demo_root_np
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=(14, 6))
            ax = fig.add_subplot(121, projection="3d")
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "-", color="tab:blue", lw=2, label=f"GT demo (RL) [{len(gt)}]")
            ax.plot(il[:, 0], il[:, 1], il[:, 2], "-", color="tab:orange", lw=2, label=f"IL fish [{len(il)}]")
            ax.scatter(*gt[0], c="green", s=80, label="start", depthshade=False)
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title("3D trajectory"); ax.legend(fontsize=8)
            ax2 = fig.add_subplot(122)
            ax2.plot(gt[:, 0], gt[:, 1], "-", color="tab:blue", lw=2, label="GT demo (RL)")
            ax2.plot(il[:, 0], il[:, 1], "-", color="tab:orange", lw=2, label="IL fish")
            ax2.scatter(gt[0, 0], gt[0, 1], c="green", s=80, label="start")
            ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)"); ax2.set_title("top-down (xy)")
            ax2.axis("equal"); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
            fig.suptitle(f"IL fish vs GT demo trajectory ({stem})  env-local frame")
            fig.tight_layout(); fig.savefig(os.path.join(self._traj_dir, stem + ".png"), dpi=120); plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"[SalmonILWater] traj compare PNG save failed: {e}", flush=True)

    # ---- env0 reward overlay grid: WORLD-frame demo(blue) vs IL(orange) clouds + velocity &
    #      orientation(head->tail) arrows. Accumulated over an episode, saved once at episode end
    #      on the SAME schedule as the trajectory plots. ----
    def _record_overlay(self, real0, sim0, real_frame0: int, chamfer0: float, vec0=None) -> None:
        self._ov_skip = getattr(self, "_ov_skip", -1) + 1
        if (self._ov_skip % max(1, int(self.cfg.overlay_stride))) != 0:
            return
        rec = {"real_frame": int(real_frame0), "chamfer": float(chamfer0),
               "real": real0.detach().cpu().numpy(), "sim": sim0.detach().cpu().numpy()}
        if vec0 is not None:
            rec.update(vec0)
        self._ov_buf.append(rec)
        if len(self._ov_buf) > 400:          # bound memory (episodes are <=195 steps)
            self._ov_buf.pop(0)

    def _write_overlay_grid(self, ep: int) -> None:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        axes = self._ov_axes
        recs = self._ov_buf
        # sample up to overlay_window cells evenly across the episode
        nmax = max(1, int(self.cfg.overlay_window))
        if len(recs) > nmax:
            idx = np.linspace(0, len(recs) - 1, nmax).astype(int)
            recs = [recs[i] for i in idx]
        all_pts = np.concatenate([r["real"][:, axes] for r in recs] + [r["sim"][:, axes] for r in recs], axis=0)
        mins, maxs = all_pts.min(0), all_pts.max(0)
        pad = np.maximum((maxs - mins) * 0.08, 1e-3)
        xlim = (mins[0] - pad[0], maxs[0] + pad[0]); ylim = (mins[1] - pad[1], maxs[1] + pad[1])
        span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-3))
        AXLEN, VELSCALE = 0.5 * span, 0.6        # arrow scales (orientation: half the span; velocity: 0.6 s)
        ncols = min(6, max(1, int(np.ceil(np.sqrt(len(recs))))))
        nrows = int(np.ceil(len(recs) / ncols))
        fig, grid = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.8 * nrows), constrained_layout=True)
        flat = np.atleast_1d(grid).reshape(-1)

        def arrow(ax, base, vec, color, ls):
            ax.annotate("", xy=(base[0] + vec[0], base[1] + vec[1]), xytext=(base[0], base[1]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.6, linestyle=ls, alpha=0.9))

        for ax, r in zip(flat, recs):
            ax.scatter(r["real"][:, axes[0]], r["real"][:, axes[1]], s=1.0, alpha=0.35, c="#1f77b4")
            ax.scatter(r["sim"][:, axes[0]], r["sim"][:, axes[1]], s=1.0, alpha=0.35, c="#ff7f0e")
            title = f"t {r['real_frame']:03d} | Chamfer {r['chamfer']:.4f}"
            if "sim_cen" in r:
                cr, cs = r["demo_cen"][list(axes)], r["sim_cen"][list(axes)]
                # orientation (head->tail) arrows: solid ;  velocity arrows: dotted
                arrow(ax, cr, r["demo_axis"][list(axes)] * AXLEN, "#1f77b4", "-")
                arrow(ax, cs, r["sim_axis"][list(axes)] * AXLEN, "#ff7f0e", "-")
                arrow(ax, cr, r["demo_vel"][list(axes)] * VELSCALE, "#1f77b4", ":")
                arrow(ax, cs, r["sim_vel"][list(axes)] * VELSCALE, "#ff7f0e", ":")
            ax.set_title(title, fontsize=7)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
        for ax in flat[len(recs):]:
            ax.axis("off")
        lbl = {0: "x", 1: "y", 2: "z"}
        fig.suptitle(f"episode {ep}  ({lbl[axes[0]]}-{lbl[axes[1]]}, env-local) -- demo(blue) vs IL(orange) | "
                     f"solid=head->tail axis, dotted=velocity", fontsize=12)
        path = os.path.join(self._overlay_dir, f"overlay_ep{ep:04d}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
