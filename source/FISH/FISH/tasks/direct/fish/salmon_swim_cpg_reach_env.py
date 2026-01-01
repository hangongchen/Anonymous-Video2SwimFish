# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""CPG+RL BASELINE for the random-target head-first reaching task.

Controlled comparison against the PCA+RL policy: the ONLY methodological difference is the
locomotion prior --

    PCA+RL:  action [da1,da2] -> PCA coefficients -> curvature -> calibrated joint decoder
    CPG+RL:  action [dA,df,dbias] -> traveling-wave CPG -> joint targets      (this file)

EVERYTHING else is inherited from SalmonSwimPCAReachEnv / the head-first cfg unchanged:
targets (front cone, normal distance, 0.2 m radius, reach->resample-no-reset), the REWARD
(bit-identical: heading-gated progress + motion-coupled alignment + alignment-gated success
bonus + joint-velocity energy term -- none of it references PCA quantities; the a1/a2 entries
in the parent's logs are logging-only and are superseded by cpg/* keys here), terminations,
episode length, FEM/water/asset/dt, and the PPO yaml (copied verbatim under a new name).

The CPG mirrors the VALIDATED scripted swimmer (scripted_swim_test / eval_pca_swim baseline,
+0.55 BL/s): q_k(t) = A * envelope(k) * sin(theta - phase_k) + bias, with theta' = 2*pi*f,
phase_k = 2*pi * rank_headfirst(k) / (N-1) (one full wave over the body, crest traveling
head->tail = forward thrust; the sign convention was measured, the + variant swims backward),
uniform envelope (the best-performing validated choice), and TURNING as a uniform joint-angle
offset (constant body curvature -> constant-radius turn) -- the minimal low-dimensional
steering modulation, not a bespoke steering controller.

Observation: identical to the PCA task except the controller state -- the PCA policy sees its
2 normalized coefficients; the CPG policy sees its 3 normalized parameters + oscillator phase
(sin/cos) which is strictly necessary to phase-coordinate turns. 32 -> 35 dims.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils

from .salmon_swim_pca_reach_env import SalmonSwimPCAReachEnv
from .salmon_swim_pca_reach_cfg import SalmonSwimCPGReachCfg


class SalmonSwimCPGReachEnv(SalmonSwimPCAReachEnv):
    cfg: SalmonSwimCPGReachCfg

    # ------------------------------------------------------------------ spaces
    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        nj = int(self._control_joint_ids.shape[0])
        self._num_cpg = 3
        # base proprioception (identical to PCA task) + CPG state (3 params + sin/cos phase)
        # + target dir_b (3) + dist (1)
        self._obs_dim = 3 + 3 + 2 * nj + 3 + 3 + (self._num_cpg + 2) + 4
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_cpg,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_cpg), device=self.device)
        print(f"[SalmonSwimCPGReach] action = 3 CPG param rates [dA, df, dbias]; "
              f"obs = {self._obs_dim} dims", flush=True)

    def __init__(self, cfg: SalmonSwimCPGReachCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        E = self.num_envs
        c = cfg
        nj = int(self._control_joint_ids.shape[0])
        # head-first rank of each controlled joint -> spatial phase (reuses the calibration npz's
        # verified head-first ':1' joint ordering, same source the PCA decoder uses)
        calib = np.load(c.pca_calib_path)
        names = [str(n) for n in calib["joint_names"]]
        phi_names = [names[i] for i in calib["fam1"]]                  # head-first
        env_names = [self.robot.data.joint_names[i] for i in self._control_joint_ids]
        rank = np.array([phi_names.index(n) for n in env_names], dtype=np.float64)
        self._cpg_phase_k = torch.tensor(
            2.0 * math.pi * float(c.cpg_wavelengths) * rank / max(1, nj - 1),
            device=dev, dtype=torch.float32)
        self._cpg_env_k = torch.ones(nj, device=dev)                   # uniform envelope (validated)
        # parameter state [A, f, bias] + oscillator phase
        self._cpg_p = torch.zeros(E, 3, device=dev)
        self._cpg_p[:, 1] = float(c.cpg_freq_init)
        self._cpg_theta = torch.zeros(E, device=dev)
        self._p_lo = torch.tensor([0.0, c.cpg_freq_range[0], -c.cpg_bias_max], device=dev)
        self._p_hi = torch.tensor([c.cpg_amp_max, c.cpg_freq_range[1], c.cpg_bias_max], device=dev)
        self._dp_max = torch.tensor([c.cpg_amp_rate, c.cpg_freq_rate, c.cpg_bias_rate], device=dev)
        self._ctrl_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        print(f"[SalmonSwimCPGReach] A in [0,{math.degrees(c.cpg_amp_max):.0f}]deg "
              f"f in {c.cpg_freq_range} Hz bias +/-{math.degrees(c.cpg_bias_max):.0f}deg | "
              f"rates/step [{math.degrees(c.cpg_amp_rate):.1f}deg, {c.cpg_freq_rate}Hz, "
              f"{math.degrees(c.cpg_bias_rate):.1f}deg]", flush=True)

    # ------------------------------------------------------------------ control
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.view(self.num_envs, self._num_cpg)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        self.actions = actions
        self._cpg_p = torch.clamp(self._cpg_p + actions * self._dp_max, self._p_lo, self._p_hi)
        A, f, bias = self._cpg_p[:, 0], self._cpg_p[:, 1], self._cpg_p[:, 2]
        # AMPLITUDE BUDGET: A_eff = min(A, limit - |bias|) so oscillation + turn offset never
        # exceeds the joint range. Without it, A=36deg + bias=25deg commands 61deg -> hard clip
        # at 45deg -> a distorted asymmetric wave that tripped the FEM guard in the smoke test.
        A = torch.minimum(A, (float(self.cfg.cpg_amp_max) - bias.abs()).clamp(min=0.0))
        self._cpg_theta = torch.remainder(
            self._cpg_theta + 2.0 * math.pi * f * self._ctrl_dt, 2.0 * math.pi)
        q = (A.unsqueeze(1) * self._cpg_env_k.unsqueeze(0)
             * torch.sin(self._cpg_theta.unsqueeze(1) - self._cpg_phase_k.unsqueeze(0))
             + bias.unsqueeze(1))
        lo = self._soft_joint_limits[:, self._control_joint_ids, 0]
        hi = self._soft_joint_limits[:, self._control_joint_ids, 1]
        self._pos_targets = torch.clamp(q, lo, hi)

    # ------------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        # same structure as the PCA reach task; controller state block swapped (documented diff)
        if self._success_markers is not None:
            self._success_markers.visualize(
                translations=self.target_positions_w,
                scales=torch.full((self.num_envs, 3), float(self._cur_radius), device=self.device))
        d = self.robot.data
        root = d.root_state_w
        fwd_b = torch.tensor([self._fwd_sign, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        heading_dir = math_utils.quat_apply(root[:, 3:7], fwd_b)
        heading_dir = heading_dir / heading_dir.norm(dim=1, keepdim=True).clamp(min=1e-6)
        up_b = torch.tensor([0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        up_dir = math_utils.quat_apply(root[:, 3:7], up_b)
        jp = d.joint_pos[:, self._control_joint_ids] * self.cfg.obs_scales.joint_pos
        jv = d.joint_vel[:, self._control_joint_ids] * self.cfg.obs_scales.joint_vel
        p_norm = (self._cpg_p - self._p_lo) / (self._p_hi - self._p_lo).clamp(min=1e-6) * 2.0 - 1.0
        delta_w = self.target_positions_w[:, 0:3] - root[:, 0:3]
        delta_b = math_utils.quat_apply_inverse(root[:, 3:7], delta_w)
        dist = delta_b.norm(dim=1, keepdim=True)
        dir_b = delta_b / dist.clamp(min=1e-6)
        obs = torch.cat((
            d.root_lin_vel_b * self.cfg.obs_scales.root_lin_vel,
            d.root_ang_vel_b * self.cfg.obs_scales.root_ang_vel,
            jp, jv, heading_dir, up_dir,
            p_norm, torch.sin(self._cpg_theta).unsqueeze(1), torch.cos(self._cpg_theta).unsqueeze(1),
            dir_b, dist,
        ), dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)}

    # ------------------------------------------------------------------ reward: INHERITED VERBATIM
    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()                # the EXACT head-first reach reward
        self.extras["log"].update({
            "cpg/A_deg": torch.rad2deg(self._cpg_p[:, 0]).mean().detach(),
            "cpg/f_hz": self._cpg_p[:, 1].mean().detach(),
            "cpg/bias_deg": torch.rad2deg(self._cpg_p[:, 2]).mean().detach(),
            "cpg/p_sat_frac": (((self._cpg_p - self._p_lo).abs() < 1e-6)
                               | ((self._cpg_p - self._p_hi).abs() < 1e-6)).float().mean().detach(),
        })
        return reward

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        if hasattr(self, "_cpg_p"):
            self._cpg_p[idx] = 0.0
            self._cpg_p[idx, 1] = float(self.cfg.cpg_freq_init)
            self._cpg_theta[idx] = 0.0
