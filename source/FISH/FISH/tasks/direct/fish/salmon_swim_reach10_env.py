# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""10-target benchmark environments: all-PC PCA vs ZeF-calibrated CPG (see the cfg file for
the protocol). TenTargetMixin owns the attempt state machine; the two env classes differ only
in the controller."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform

from .salmon_swim_pca_reach_env import SalmonSwimPCAReachEnv
from .salmon_swim_reach10_cfg import SalmonSwimCPGReach10Cfg, SalmonSwimPCAReach10Cfg


class TenTargetMixin:
    """Attempt state machine: success / 20 s timeout / early abort -> record 0/1, next target,
    NO reset; episode truncates after exactly n_targets_per_episode attempts."""

    def _init_ten_target(self):
        E, dev = self.num_envs, self.device
        c = self.cfg
        self._n_att = int(c.n_targets_per_episode)
        self._att_timeout_steps = int(round(float(c.attempt_timeout_s)
                                            / (self.cfg.sim.dt * self.cfg.decimation)))
        self._att_idx = torch.zeros(E, dtype=torch.long, device=dev)
        self._att_timer = torch.zeros(E, dtype=torch.long, device=dev)
        self._att_results = torch.full((E, self._n_att), -1, dtype=torch.long, device=dev)
        self._ep_outcomes = deque(maxlen=200)          # per-episode result vectors (ended eps)
        # optional deterministic RELATIVE target sequences (evaluation): (E, n_att, 2)
        self._tgt_seq = None
        sp = str(getattr(c, "target_seq_path", "") or "")
        if sp:
            z = np.load(sp)
            off, dist = z["bearing_off"], z["dist"]
            assert off.shape[0] >= E and off.shape[1] >= self._n_att, \
                f"target seq {off.shape} too small for {E} envs x {self._n_att} attempts"
            self._tgt_seq = (torch.tensor(off[:E, :self._n_att], device=dev, dtype=torch.float32),
                             torch.tensor(dist[:E, :self._n_att], device=dev, dtype=torch.float32))
            print(f"[Reach10] DETERMINISTIC target sequences loaded from {sp}", flush=True)
        print(f"[Reach10] {self._n_att} attempts/episode, timeout {self._att_timeout_steps} steps, "
              f"abort dist {c.attempt_abort_dist} m", flush=True)

    # ---- target sampling: sequence-driven when a sequence is loaded ----
    def _sample_targets(self, env_ids, base_pos, heading_az=None):
        if self._tgt_seq is None:
            return super()._sample_targets(env_ids, base_pos, heading_az)
        ids = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, device=self.device)
        n = int(len(ids))
        if heading_az is None:
            rq = self.robot.data.root_state_w[ids, 3:7]
            fs = float(getattr(self.cfg, "body_forward_sign", -1.0))
            fwd_b = torch.tensor([fs, 0.0, 0.0], dtype=rq.dtype, device=rq.device).repeat(n, 1)
            fwd_w = math_utils.quat_apply(rq, fwd_b)
            heading_az = torch.atan2(fwd_w[:, 1], fwd_w[:, 0])
        att = self._att_idx[ids].clamp(max=self._n_att - 1)
        bearing = heading_az.to(self.device) + self._tgt_seq[0][ids, att]
        dist = self._tgt_seq[1][ids, att]
        tgt = base_pos.clone().to(self.device)
        tgt[:, 0] = base_pos[:, 0] + dist * torch.cos(bearing)
        tgt[:, 1] = base_pos[:, 1] + dist * torch.sin(bearing)
        tgt[:, 2] = self._target_height
        self.target_positions_w[ids] = tgt

    # ---- the attempt state machine replaces the fixed-horizon dones ----
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        root_state = self.robot.data.root_state_w
        local_pos = root_state[:, 0:3] - self.scene.env_origins
        out_of_bounds = torch.linalg.norm(local_pos, dim=1) > self.cfg.root_position_limit
        joint_blowup = self.joint_vel.abs().max(dim=1).values > self.cfg.joint_velocity_limit
        nonfinite = (~torch.isfinite(root_state).all(dim=1)) \
            | (~torch.isfinite(self.joint_pos).all(dim=1)) | (~torch.isfinite(self.joint_vel).all(dim=1))
        self._fem_blowup = torch.zeros_like(joint_blowup)
        _fvl = getattr(self.cfg, "fem_velocity_limit", None)
        if _fvl is not None and getattr(self, "_soft_view", None) is not None:
            try:
                _nvel = self._soft_view.get_simulation_mesh_nodal_velocities()
                _nvmax = torch.nan_to_num(_nvel, nan=float("inf")).abs().flatten(1).amax(dim=1)
                self._fem_blowup = _nvmax > float(_fvl)
            except Exception:  # noqa: BLE001
                pass
        nonfinite = nonfinite | self._fem_blowup
        _blew = joint_blowup | nonfinite
        self._blew_now = int(_blew.sum())
        _jvm = self.joint_vel.abs().max()
        self._jvel_max = float(_jvm) if torch.isfinite(_jvm) else float("inf")
        terminated = out_of_bounds | joint_blowup | nonfinite

        dist = torch.linalg.norm(self.target_positions_w[:, 0:3] - root_state[:, 0:3], dim=1)
        self._att_timer += 1
        success = dist < self._cur_radius
        timeout = self._att_timer >= self._att_timeout_steps
        abort = dist > float(self.cfg.attempt_abort_dist)
        advance = (success | timeout | abort) & ~terminated
        self._reached_now = success & advance                      # reward's success bonus
        self._targets_reached += self._reached_now.float()

        if advance.any():
            ids = advance.nonzero(as_tuple=False).flatten()
            self._att_results[ids, self._att_idx[ids].clamp(max=self._n_att - 1)] = \
                success[ids].long()
            self._att_idx[ids] += 1
            cont = ids[self._att_idx[ids] < self._n_att]
            if len(cont) > 0:
                self._sample_targets(cont, root_state[cont, 0:3])
                self._att_timer[cont] = 0
                self._prev_distance[cont] = torch.linalg.norm(
                    self.target_positions_w[cont, 0:3] - root_state[cont, 0:3], dim=1)

        episode_done = self._att_idx >= self._n_att
        # backstop only (protocol ends first: 10 x timeout <= episode_length_s)
        episode_done = episode_done | (self.episode_length_buf >= self.max_episode_length - 1)
        self.extras.setdefault("log", {}).update({
            "reach10/att_idx": self._att_idx.float().mean().detach(),
            "reach10/succ_frac_running": (self._att_results.clamp(min=0).sum(1).float()
                                          / self._att_idx.clamp(min=1).float()).mean().detach(),
            "reach10/targets_per_ep": torch.tensor(
                float(np.mean([o.sum() for o in self._ep_outcomes])) if self._ep_outcomes else 0.0),
            "reach10/blowups": _blew.float().sum().detach(),
        })
        self.extras["success"] = self._reached_now.detach().clone()
        return terminated, episode_done

    def _reset_idx(self, env_ids: Sequence[int] | None):
        ids = self.robot._ALL_INDICES if env_ids is None else env_ids
        lst = ids.tolist() if torch.is_tensor(ids) else list(ids)
        if hasattr(self, "_att_results"):
            for e in lst:
                res = self._att_results[e]
                if (res >= 0).any():                      # completed(ish) episode -> record
                    self._ep_outcomes.append(res.clamp(min=0).cpu().numpy())
            self._att_idx[lst] = 0
            self._att_timer[lst] = 0
            self._att_results[lst] = -1
        super()._reset_idx(env_ids)


class SalmonSwimPCAReach10Env(TenTargetMixin, SalmonSwimPCAReachEnv):
    """All-PC PCA controller on the 10-target protocol (action = 20 coefficient rates)."""

    cfg: SalmonSwimPCAReach10Cfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_ten_target()


class SalmonSwimCPGReach10Env(SalmonSwimPCAReach10Env):
    """ZeF-calibrated CPG on the 10-target protocol.

    kappa(s,t) = kappa_mean(s) + A*E(s)*sin(theta + phi(s)) + b   ->   q = Mdec (kappa - k_rest)
    which collapses to  q(t) = q0 + A*(sin(theta) u1 + cos(theta) u2) + b * ub  with
    u1 = M(E*cos phi), u2 = M(E*sin phi), ub = M 1.  Action = [dA, df, db], all bounds and
    rates measured from the ZeF data (cpg_params.npz). Same decoder, same everything else."""

    cfg: SalmonSwimCPGReach10Cfg

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        nj = int(self._control_joint_ids.shape[0])
        self._num_cpg = 3
        # base proprioception + CPG state (3 params + sin/cos phase) + target dir_b + dist
        self._obs_dim = 3 + 3 + 2 * nj + 3 + 3 + (self._num_cpg + 2) + 4
        self.single_observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(self._obs_dim,))})
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space["policy"], self.num_envs)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_cpg,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.actions = torch.zeros((self.num_envs, self._num_cpg), device=self.device)
        print(f"[CPGReach10] action = [dA, df, db]; obs = {self._obs_dim}", flush=True)

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        pth = str(cfg.cpg_params_path or "") or str(
            Path(__file__).resolve().parents[6] / "demo_out/zef_manifold/cpg_params.npz")
        p = np.load(pth)
        E_s = torch.tensor(p["envelope"], device=dev, dtype=torch.float32)      # (20,)
        phi = torch.tensor(p["phase"], device=dev, dtype=torch.float32)         # (20,)
        M = self._Mdec                                                          # (7, 20) shared decoder
        self._cpg_u1 = M @ (E_s * torch.cos(phi))                               # (7,)
        self._cpg_u2 = M @ (E_s * torch.sin(phi))
        self._cpg_ub = M @ torch.ones_like(E_s)
        self._cpg_f0 = float(p["f0"])
        self._cp_lo = torch.tensor([0.0, float(p["f_lo"]), -float(p["b_max"])], device=dev)
        self._cp_hi = torch.tensor([float(p["A_max"]), float(p["f_hi"]), float(p["b_max"])], device=dev)
        self._cdp = torch.tensor([float(p["dA_max"]), float(p["df_max"]), float(p["db_max"])], device=dev)
        self._cpg_p = torch.zeros(self.num_envs, 3, device=dev)
        self._cpg_p[:, 1] = self._cpg_f0
        self._cpg_theta = torch.zeros(self.num_envs, device=dev)
        self._ctrl_dt = float(self.cfg.sim.dt * self.cfg.decimation)
        print(f"[CPGReach10] ZeF-calibrated: A<= {float(p['A_max']):.2f} f in "
              f"[{float(p['f_lo']):.2f},{float(p['f_hi']):.2f}] b<= {float(p['b_max']):.2f} | "
              f"rates {self._cdp.cpu().numpy().round(3).tolist()} | f0={self._cpg_f0:.2f} Hz", flush=True)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.view(self.num_envs, self._num_cpg)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        self.actions = actions
        self._cpg_p = torch.clamp(self._cpg_p + actions * self._cdp, self._cp_lo, self._cp_hi)
        A, f, b = self._cpg_p[:, 0], self._cpg_p[:, 1], self._cpg_p[:, 2]
        self._cpg_theta = torch.remainder(
            self._cpg_theta + 2.0 * math.pi * f * self._ctrl_dt, 2.0 * math.pi)
        q = (self._q0.unsqueeze(0)
             + A.unsqueeze(1) * (torch.sin(self._cpg_theta).unsqueeze(1) * self._cpg_u1.unsqueeze(0)
                                 + torch.cos(self._cpg_theta).unsqueeze(1) * self._cpg_u2.unsqueeze(0))
             + b.unsqueeze(1) * self._cpg_ub.unsqueeze(0))
        lo = self._soft_joint_limits[:, self._control_joint_ids, 0]
        hi = self._soft_joint_limits[:, self._control_joint_ids, 1]
        self._pos_targets = torch.clamp(q, lo, hi)

    def _get_observations(self) -> dict:
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
        p_norm = (self._cpg_p - self._cp_lo) / (self._cp_hi - self._cp_lo).clamp(min=1e-6) * 2.0 - 1.0
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

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()                   # the EXACT same task reward
        self.extras["log"].update({
            "cpg/A": self._cpg_p[:, 0].mean().detach(),
            "cpg/f_hz": self._cpg_p[:, 1].mean().detach(),
            "cpg/b": self._cpg_p[:, 2].mean().detach(),
            "cpg/p_sat_frac": (((self._cpg_p - self._cp_lo).abs() < 1e-6)
                               | ((self._cpg_p - self._cp_hi).abs() < 1e-6)).float().mean().detach(),
        })
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        if hasattr(self, "_cpg_p"):
            self._cpg_p[idx] = 0.0
            self._cpg_p[idx, 1] = self._cpg_f0
            self._cpg_theta[idx] = 0.0
