#!/usr/bin/env bash
# Repeatable DEBUG-test launcher for Template-Salmon-Swim-Direct-v0.
# Same command every time: soft body ON + per-step debug prints ON, so you can watch the
# success -> reset -> soft-body-explosion sequence and read the reward terms / done reasons.
#
# Defaults: 1 env, GUI (non-headless), soft body active, recording OFF, debug prints ON.
#
# Usage:
#   bash scripts/test_salmon_swim_debug.sh                 # the standard debug test
#   NUM_ENVS=256 bash scripts/test_salmon_swim_debug.sh    # scale up
#   HEADLESS=1   bash scripts/test_salmon_swim_debug.sh    # no GUI (prints still go to stdout)
#   DEV=cuda:1   bash scripts/test_salmon_swim_debug.sh    # other GPU
#   bash scripts/test_salmon_swim_debug.sh --max_iterations 50   # extra args pass through
#   bash scripts/test_salmon_swim_debug.sh env.debug_print=false # silence the per-step prints
set -u

REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV=${DEV:-cuda:0}
NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-0}

cd "$REPO"
export WANDB_MODE=${WANDB_MODE:-online}
export HYDRA_FULL_ERROR=1
export DISPLAY=${DISPLAY:-:1}

# clean up any leftover salmon-swim training first (matches the python cmdline, NOT this
# wrapper -- the wrapper's argv has no "rl_games/train_ppo.py", so it won't self-kill).
echo "[test] cleaning up leftover salmon-swim training..."
pkill -9 -f "rl_games/train_ppo.py.*Template-Salmon-Swim-Direct-v0" 2>/dev/null || true
sleep 2

HEADLESS_FLAG=""
[ "$HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"

echo "[test] num_envs=$NUM_ENVS device=$DEV headless=$HEADLESS | soft body ON, debug prints ON"
echo "[test] watch for:  [dbg done] ... env0 RESET: SUCCESS   then next step  jvel_max -> huge (= soft body explodes on reset)"
exec "$PY" scripts/rl_games/train_ppo.py \
    --task Template-Salmon-Swim-Direct-v0 \
    --num_envs "$NUM_ENVS" --device "$DEV" $HEADLESS_FLAG \
    --target_offset_xy 1 1 \
    --track \
    --wandb-project-name fish_articulation_analytic_water \
    --wandb-entity ANON-ENTITY \
    --wandb-name salmon_swim_debug \
    env.with_deformable=true \
    env.record_video=false \
    env.debug_print=true \
    "$@"
