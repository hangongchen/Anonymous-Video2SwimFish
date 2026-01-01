#!/usr/bin/env bash
# Misty panels swim+AMP training on the RE-TUNED fish (2026-08-04).
#
# Everything below was measured with an open-loop scripted gait (scripts/scripted_swim_test.py),
# isolated 40 s runs, ZERO blow-ups at every setting tried:
#
#   water     panel_cd_tangent 0.2 -> 0.01   (0.2 was the ellipsoid model's cd_slender, a crossflow
#                                             correction, misused as wetted-area skin friction; it
#                                             made 90% of all forward drag)
#   joints    +/-15 -> +/-25 deg             (0.136 -> 0.236 BL/s; +/-35 and +/-50 are SLOWER)
#   scale     pos_action_scale -> 0.4363     (MUST match the limit in rad or nothing changes)
#   drive     stiffness 60 -> 240            (the drive is a first-order lag, corner = k/c; at a 3 Hz
#                                             beat the joints reached only 47% of command. 240 moves
#                                             the corner 1.6 -> 6.4 Hz and tracking to 91%)
#
#   scripted-gait ceiling: 0.124 -> 0.691 BL/s (5.6x), ~35% of a real fish (Bainbridge at 4 Hz = 2.0).
#
# NO WARM START. The old ep_2320 checkpoint was trained on a fish with half the joint range, a quarter
# the drive stiffness and 20x the skin drag; it already diverged when only the water changed
# (distance_to_target rose monotonically 1.37 -> 1.61 m over 16 epochs, 0 reaches). Set CKPT=<path>
# to warm start anyway.
#
# KNOWN RISK: the previous policy lost to a plain sine wave (~0 forward speed vs +0.121 BL/s), and the
# prime suspect is the AMP style reward imitating a noisy reference. If this run also fails to swim,
# the next test is the same run with the AMP weight at zero.
#
# Usage:  bash scripts/train_misty_fast.sh                 # cuda:0, 256 envs
#         DEV=cuda:1 bash scripts/train_misty_fast.sh      # other GPU
#         CKPT=logs/.../foo.pth bash scripts/train_misty_fast.sh   # warm start anyway
set -u
# NOTE env.random_target=true is REQUIRED and was the bug that broke the 2026-08-04/05 runs:
# the cfg default is False, which pins the target to a FIXED (1,1) world offset (1.414 m, 45 deg)
# for every env and every episode. Combined with initial_root_{pos,rot}_range=(0,0) that leaves the
# task with ZERO episode-level randomisation -- a memorised trajectory scores ~100%. The runs from
# 2026-07-30..2026-08-03 passed it on the CLI; train_misty_lowdrag_warmstart.sh dropped it while
# claiming to reproduce them.

REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV=${DEV:-cuda:0}
HEADLESS=${HEADLESS:-1}
HEADLESS_FLAG=""
[ "$HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"
CKPT_FLAG=""
[ -n "${CKPT:-}" ] && CKPT_FLAG="--checkpoint $CKPT"
NAME=${NAME:-misty_fast}

cd "$REPO"
export WANDB_MODE=${WANDB_MODE:-online}
export HYDRA_FULL_ERROR=1
export DISPLAY=${DISPLAY:-:1}

# simulation_app.close() hangs on this box (Blackwell + FEM) -- kill leftovers by PID, and kill the
# PYTHON child, not just a `timeout`/shell wrapper.
exec "$PY" scripts/rl_games/train_ppo.py \
    --task Template-Salmon-Swim-AMP-Misty-Panels-Direct-v0 \
    --num_envs 256 --device "$DEV" $HEADLESS_FLAG $CKPT_FLAG \
    --track \
    --wandb-project-name fish_articulation_analytic_water \
    --wandb-entity ANON-ENTITY \
    --wandb-name "$NAME" \
    env.random_target=true \
    env.success_radius=0.4 \
    env.episode_length_s=200.0 \
    env.dist_curriculum=true \
    env.dist_curriculum_gated=true \
    env.dist_curriculum_start=0.8 \
    env.debug_print=true \
    "$@"
