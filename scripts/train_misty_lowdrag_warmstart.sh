#!/usr/bin/env bash
# Misty panels swim+AMP, WARM-STARTED from the pre-collapse swimmer, in the CORRECTED water.
#
# Why this run exists:
#   * water fix -- panel_cd_tangent 0.2 -> 0.02 (cfg default now). The old value was copied from the
#     ellipsoid model's cd_slender, where it means a crossflow correction, not a wetted-area skin
#     friction; at 0.2 it produced 90% of all forward drag and the fish could not coast at all
#     (glide 0.58 BL). Measured after the fix: glide 2.81 BL, inside the real-fish band 1.8-3.6 BL.
#   * warm start -- ep_2320 of run 2026-08-03_01-10-46 is the strongest swimmer we have (it sits in
#     the 0.41 gate-rate window, before that run's mid-training reach-skill collapse). The gait is
#     still useful; only the push-per-beat changed, so re-learning it from scratch is waste.
#     NOTE: an A/B (same policy, old vs new water) showed the old policy does NOT speed up by itself
#     -- it maneuvers instead of cruising, so the gain has to be learned. That is this run's job.
#
# Everything else reproduces the gated run exactly (settings read back from its params/env.yaml):
#   pos_action_scale 0.26 (max USEFUL: the joint saturates at +/-15 deg; sweep found no blow-up
#   break point up to 0.50), success_radius 0.4, episode 200 s, rate-GATED distance curriculum
#   from 0.8 m (promote at reach-rate 0.20, demote at 0.05 -- a cumulative counter promotes a
#   FAILING policy, a rate gate does not).
#
# Usage:  bash scripts/train_misty_lowdrag_warmstart.sh                 # cuda:0, 256 envs
#         DEV=cuda:1 bash scripts/train_misty_lowdrag_warmstart.sh      # other GPU
#         bash scripts/train_misty_lowdrag_warmstart.sh --num_envs 512  # extra args pass through
set -u

REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV=${DEV:-cuda:0}
CKPT=${CKPT:-logs/rl_games/cartpole_direct/2026-08-03_01-10-46/nn/last_cartpole_direct_ep_2320_rew_1152.9531.pth}
HEADLESS=${HEADLESS:-1}
HEADLESS_FLAG=""
[ "$HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"

cd "$REPO"
export WANDB_MODE=${WANDB_MODE:-online}
export HYDRA_FULL_ERROR=1
export DISPLAY=${DISPLAY:-:1}

# NOTE: simulation_app.close() hangs on this box (Blackwell + FEM) -- kill leftovers by PID, and
# kill the PYTHON child, not just a `timeout` wrapper.
exec "$PY" scripts/rl_games/train_ppo.py \
    --task Template-Salmon-Swim-AMP-Misty-Panels-Direct-v0 \
    --checkpoint "$CKPT" \
    --num_envs 256 --device "$DEV" $HEADLESS_FLAG \
    --track \
    --wandb-project-name fish_articulation_analytic_water \
    --wandb-entity ANON-ENTITY \
    --wandb-name misty_lowdrag_warmstart \
    env.pos_action_scale=0.26 \
    env.success_radius=0.4 \
    env.episode_length_s=200.0 \
    env.dist_curriculum=true \
    env.dist_curriculum_gated=true \
    env.dist_curriculum_start=0.8 \
    env.debug_print=true \
    "$@"
