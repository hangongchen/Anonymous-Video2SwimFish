# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stress-test cfg: the VLM/auto-skeleton-generated fish (auto_skeleton_fish/fish_articulated.usd)
in the tank-swim task, but at Isaac Sim's DEFAULT dt (1/60) instead of the FEM-stable 1/960.

Purpose: verify the blow-up check by running a config that SHOULD blow up. Per the project's own CFL
findings (1/240, 1/480, 1/720 all blew up; only 1/960 stable), a coarse 1/60 dt is far above the FEM
CFL ceiling, so the cooked deformable should diverge quickly. `fem_velocity_limit` is enabled so the
guard catches a soft-body divergence even if it precedes the joint blow-up.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .salmon_swim_cfg import SALMON_SWIM_ARTICULATION_CFG
from .salmon_tank_swim_cfg import SalmonTankSwimEnvCfg

_NEW_USD = (
    Path(__file__).resolve().parents[6]
    / "SimFishLib" / "fish_asset_pipeline" / "generated_usd_dataset"
    / "auto_skeleton_fish" / "fish_articulated.usd"
)

_ART: ArticulationCfg = copy.deepcopy(SALMON_SWIM_ARTICULATION_CFG)
_ART.spawn.usd_path = str(_NEW_USD)
_ART.init_state = ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0))

# dt is env-configurable so the same task can sweep physics rates. Default 60 Hz = Isaac's default dt.
# A positive-control blow-up run uses a much coarser rate (e.g. FISH_TEST_DT_HZ=15) where the FEM CFL
# diverges from numerical noise alone (control rate is irrelevant for the blow-up test).
_HZ = float(os.environ.get("FISH_TEST_DT_HZ", "60"))
_DEFAULT_DT = 1.0 / _HZ
_DECIM = max(1, round(_HZ / 30.0))   # ~30 Hz control when possible


@configclass
class SalmonAutoskelDefaultDtCfg(SalmonTankSwimEnvCfg):
    robot_cfg: ArticulationCfg = _ART

    # ---- the whole point: DEFAULT dt, not the CFL-stable 1/960 ----
    decimation = _DECIM
    episode_length_s = 1500 * _DECIM * _DEFAULT_DT      # keep 1500 control steps
    sim: SimulationCfg = SimulationCfg(
        dt=_DEFAULT_DT, render_interval=_DECIM, gravity=(0.0, 0.0, 0.0),
        physx=PhysxCfg(
            solver_type=1, enable_external_forces_every_iteration=True,
            gpu_collision_stack_size=2 ** 30, gpu_heap_capacity=2 ** 30,
            gpu_temp_buffer_capacity=2 ** 28, gpu_max_soft_body_contacts=2 ** 22,
            gpu_max_particle_contacts=2 ** 28,
        ),
    )

    # enable the FEM soft-body blow-up guard for this test (m/s)
    fem_velocity_limit = 50.0

    # FEM stiffness is env-configurable: FISH_TEST_YOUNGS=5e7 restores the documented CFL-UNSTABLE
    # stiffness (the value that blew up before the 1e5 softening cure) -> a guaranteed positive control
    # for the blow-up detector. Default 1e5 (the stable cure).
    fem_youngs_modulus = float(os.environ.get("FISH_TEST_YOUNGS", "1e5"))

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=8.0, replicate_physics=False,
    )
    debug_print = True
