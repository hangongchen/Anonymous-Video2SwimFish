# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configs for the 10-TARGET controlled benchmark: ZeF-calibrated CPG+RL vs all-PC PCA+RL.

PROTOCOL (identical for both methods; implemented in salmon_swim_reach10_env.TenTargetMixin):
  - each episode = EXACTLY 10 target attempts; the attempt index is NOT in the observation
  - per attempt: reach (dist < 0.2 m) -> success=1; 20 s timeout -> 0; early abort
    (dist > attempt_abort_dist, fixed+documented, same for both methods) -> 0
  - every outcome IMMEDIATELY samples the next target from the fish's current pose
    (no reset, no episode termination); episode ends only after 10 attempts (max 200 s)
  - primary metric: targets_reached / 10
  - targets: same rules as before (uniform +/-15 deg cone about current heading, distance
    ~ N(1.0, 0.25) clamped [0.4, 1.8], radius 0.2); for EVALUATION a seeded per-(env,attempt)
    RELATIVE sequence file makes both methods face identical (bearing-offset, distance) draws
  - reward: the head-first reach reward, unchanged, same weights, no method-specific terms

The two cfgs below differ ONLY in the controller block (locomotion representation).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .salmon_swim_pca_reach_cfg import SalmonSwimPCAReachHeadCfg


@configclass
class _Reach10ProtocolCfg(SalmonSwimPCAReachHeadCfg):
    """Shared 10-target protocol fields (both methods inherit these unchanged)."""

    n_targets_per_episode = 10
    attempt_timeout_s = 20.0
    # EARLY ABORT: target is unrecoverable if the fish gets this far from it. Targets spawn at
    # <= 1.8 m; drifting past 3.0 m cannot be recovered within the 20 s budget at these swim
    # speeds. FIXED and identical for both methods -- never tuned per controller.
    attempt_abort_dist = 3.0
    # episode backstop; the protocol itself ends episodes at 10 attempts (<= 200 s)
    episode_length_s = 205.0
    # deterministic RELATIVE target sequences for evaluation ("" = sample live, training)
    target_seq_path = ""
    # the old fixed-length multi-target machinery must not interfere: the mixin owns dones
    multi_target = True                      # (kept True: reward/marker helpers expect it)


@configclass
class SalmonSwimPCAReach10Cfg(_Reach10ProtocolCfg):
    """All-PC PCA controller: action = bounded-rate deltas on ALL 20 PCA coefficients."""

    pca_num_modes = 20                       # THE change vs the old experiment: full basis


@configclass
class SalmonSwimCPGReach10Cfg(_Reach10ProtocolCfg):
    """ZeF-calibrated CPG controller (see salmon_swim_reach10_env.SalmonSwimCPGReach10Env).

    kappa_cpg(s,t) = kappa_mean(s) + A(t) * E(s) * sin(theta(t) + phi(s)) + b(t)
    with E(s), phi(s), f0/f-range, A/b bounds AND rate bounds all measured from the SAME
    ZeF-05 curvature data as the PCA basis (scripts/zef_pca_rl/calibrate_cpg_from_zef.py ->
    demo_out/zef_manifold/cpg_params.npz). Joints via the IDENTICAL ridge decoder as PCA.
    RL controls [dA, df, db] (3-D)."""

    cpg_params_path = ""                     # "" -> <repo>/demo_out/zef_manifold/cpg_params.npz
