#!/usr/bin/env bash
# Single-GPU launcher for Template-Salmon-IL-Water-Direct-v0:
# the deformable salmon learns to IMITATE a real fish's swimming gait (negative Chamfer
# distance between the simulated FEM-fish point cloud and a real-fish video point-cloud
# frame) under the analytic MuJoCo water model. Physics is the proven-stable swim config
# (dt = 1/960, decimation 32, FEM body active, FEM material cure). The FEM body is ALWAYS
# active here (the reward reads the deformable mesh), so this is the heavier "soft" path.
#
# Multi-GPU is unavailable on this box (NCCL illegal-memory-access), so single-GPU only.
#
# Usage:
#   bash scripts/train_salmon_IL_water.sh                   # cuda:0, 64 envs, GUI off
#   DEV=cuda:1 bash scripts/train_salmon_IL_water.sh        # other GPU / 2nd run
#   bash scripts/train_salmon_IL_water.sh --num_envs 128    # scale up
#   HEADLESS=0 bash scripts/train_salmon_IL_water.sh --num_envs 4   # watch a few fish
set -u

REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV=${DEV:-cuda:0}
NUM_ENVS=${NUM_ENVS:-64}
# HEADLESS=1 (default): no GUI (fast). HEADLESS=0: open the Isaac Sim viewport.
HEADLESS=${HEADLESS:-1}
HEADLESS_FLAG=""
[ "$HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"

echo "[launch] cleaning up leftover salmon-IL-water training..."
pkill -9 -f "Template-Salmon-IL-Water-Direct-v0.*--headless" 2>/dev/null || true
pkill -9 -f "train_ppo.py.*--distributed" 2>/dev/null || true
sleep 2

cd "$REPO"
export WANDB_MODE=${WANDB_MODE:-online}
export HYDRA_FULL_ERROR=1
export DISPLAY=${DISPLAY:-:1}

exec "$PY" scripts/rl_games/train_ppo.py \
    --task Template-Salmon-IL-Water-Direct-v0 \
    --num_envs "$NUM_ENVS" --device "$DEV" $HEADLESS_FLAG \
    --track \
    --wandb-project-name fish_articulation_analytic_water \
    --wandb-entity ANON-ENTITY \
    --wandb-name salmon_IL_water \
    "$@"
