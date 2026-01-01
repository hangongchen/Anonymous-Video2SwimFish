# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Fish-Direct-v0",
    entry_point=f"{__name__}.fish_env:FishEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fish_env_cfg:FishEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

gym.register(
    id="Template-Salmon-IL-Direct-v0",
    entry_point=f"{__name__}.salmon_IL:SalmonILEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_IL_cfg:SalmonILEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

gym.register(
    id="Template-Salmon-Swim-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_env:SalmonSwimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_cfg:SalmonSwimEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

gym.register(
    id="Template-Salmon-IL-Water-Direct-v0",
    entry_point=f"{__name__}.salmon_IL_water_env:SalmonILWaterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_IL_water_cfg:SalmonILWaterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# Reactive natural-swimming AMP tank task (cooked zebrafish, walls, env-owned discriminator).
gym.register(
    id="Template-Salmon-AMP-Tank-Direct-v0",
    entry_point=f"{__name__}.salmon_amp_tank_env:SalmonAMPTankEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_amp_tank_cfg:SalmonAMPTankEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# Tank-swim: cooked zebrafish in a square tank with four collidable walls (AMP pipeline step 1).
gym.register(
    id="Template-Salmon-Tank-Swim-Direct-v0",
    entry_point=f"{__name__}.salmon_tank_swim_env:SalmonTankSwimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_tank_swim_cfg:SalmonTankSwimEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# AMP reactive-swimming on the auto-skeleton fish at DEFAULT dt (1/60).
gym.register(
    id="Template-Salmon-AMP-Autoskel-Direct-v0",
    entry_point=f"{__name__}.salmon_amp_tank_env:SalmonAMPTankEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_amp_autoskel_cfg:SalmonAMPAutoskelCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# Blow-up stress test: the VLM/auto-skeleton-generated fish at Isaac's DEFAULT dt (1/60, not 1/960).
gym.register(
    id="Template-Salmon-Autoskel-DefaultDt-Direct-v0",
    entry_point=f"{__name__}.salmon_tank_swim_env:SalmonTankSwimEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_autoskel_defaultdt_cfg:SalmonAutoskelDefaultDtCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# OLD swim-to-target task (SalmonSwimEnv reward + obs, WHOLE) + an added AMP style term, on the
# auto-skeleton fish at dt=1/120. reward = old_swim_reward + w_amp * r_style.
gym.register(
    id="Template-Salmon-Swim-AMP-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_amp_env:SalmonSwimAMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_amp_cfg:SalmonSwimAMPCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# Swim-to-target + AMP on the Meshy 'Misty Minnow' fish (finned, head/tail-corrected pipeline output).
gym.register(
    id="Template-Salmon-Swim-AMP-Misty-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_amp_env:SalmonSwimAMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_amp_misty_cfg:SalmonSwimAMPMistyCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# Training 2: Misty swim+AMP but water force from the .obj SURFACE PANELS (strip theory), not ellipsoids.
gym.register(
    id="Template-Salmon-Swim-AMP-Misty-Panels-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_amp_env:SalmonSwimAMPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_amp_misty_panels_cfg:SalmonSwimAMPMistyPanelsCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# DECISIVE 2-PC EXPERIMENT: PPO in the real-fish PCA locomotion-manifold action space
# (a1,a2 -> curvature -> calibrated joint map), straight-forward-swim task, NO target.
# Separate task id + separate agent yaml so the AMP experiments stay reproducible.
gym.register(
    id="Template-Salmon-Swim-PCA-Misty-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_pca_env:SalmonSwimPCAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_pca_cfg:SalmonSwimPCACfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pca_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# PCA-manifold RANDOM-TARGET REACHING: multi-reach episodes (reach -> resample, no reset),
# front-cone normal-distance targets, progress+success reward. Own agent yaml.
gym.register(
    id="Template-Salmon-Swim-PCA-Reach-Misty-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_pca_reach_env:SalmonSwimPCAReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_pca_reach_cfg:SalmonSwimPCAReachCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pca_reach_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# HEAD-FIRST reaching: heading-gated progress + motion-coupled alignment (closes the
# broadside-gliding loophole). Same env class; cfg turns the gate on. v1 task untouched.
gym.register(
    id="Template-Salmon-Swim-PCA-Reach-Head-Misty-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_pca_reach_env:SalmonSwimPCAReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_pca_reach_cfg:SalmonSwimPCAReachHeadCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_pca_reach_head_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# CPG+RL BASELINE: identical head-first reach task; controller = traveling-wave CPG
# [amplitude, frequency, turn_bias] instead of the PCA manifold. Controlled comparison.
gym.register(
    id="Template-Salmon-Swim-CPG-Reach-Misty-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_cpg_reach_env:SalmonSwimCPGReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_pca_reach_cfg:SalmonSwimCPGReachCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cpg_reach_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# 10-TARGET CONTROLLED BENCHMARK: all-PC PCA vs ZeF-calibrated CPG, identical protocol.
gym.register(
    id="Template-Salmon-Reach10-PCA-Misty-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_reach10_env:SalmonSwimPCAReach10Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_reach10_cfg:SalmonSwimPCAReach10Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_reach10_pca_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)
gym.register(
    id="Template-Salmon-Reach10-CPG-Misty-Direct-v0",
    entry_point=f"{__name__}.salmon_swim_reach10_env:SalmonSwimCPGReach10Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_swim_reach10_cfg:SalmonSwimCPGReach10Cfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_reach10_cpg_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)

# Smoke-test variant: SimFishLib zebrafish agent + real 3D-ZeF swimming-track demo.
gym.register(
    id="Template-Salmon-IL-Water-ZeF-Direct-v0",
    entry_point=f"{__name__}.salmon_IL_water_env:SalmonILWaterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.salmon_IL_water_zef_cfg:SalmonILWaterZefEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_sac_cfg_entry_point": f"{agents.__name__}:rl_games_sac_cfg.yaml",
    },
)
