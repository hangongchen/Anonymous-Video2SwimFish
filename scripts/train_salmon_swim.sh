#!/usr/bin/env bash
# Reliable SINGLE-GPU launcher for Template-Salmon-Swim-Direct-v0.
#
# Multi-GPU is NOT usable on this box: NCCL throws "CUDA illegal memory access" in the
# first cross-GPU broadcast even with NCCL_P2P_DISABLE=1 (Blackwell + this torch/NCCL).
# Single GPU avoids NCCL entirely and 1024-2048 envs on one RTX PRO 6000 is plenty fast.
#
# Usage:
#   bash scripts/train_salmon_swim.sh                       # cuda:0, headless, 1024 envs
#   DEV=cuda:1 bash scripts/train_salmon_swim.sh            # train on the other GPU
#   bash scripts/train_salmon_swim.sh --num_envs 2048       # extra args pass through
set -u

REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV=${DEV:-cuda:0}

echo "[launch] cleaning up leftover HEADLESS salmon-swim training (leaves any GUI viewer)..."
pkill -9 -f "Template-Salmon-Swim-Direct-v0.*--headless" 2>/dev/null || true
pkill -9 -f "train_ppo.py.*--distributed" 2>/dev/null || true
sleep 2

cd "$REPO"
export WANDB_MODE=${WANDB_MODE:-online}
export HYDRA_FULL_ERROR=1
export DISPLAY=${DISPLAY:-:1}

# NOTE: wandb entity is ANON-ENTITY. If this box is logged in to a
# different wandb account it syncs OFFLINE; `wandb login --relogin` to an authorized
# account for online sync.
exec "$PY" scripts/rl_games/train_ppo.py \
    --task Template-Salmon-Swim-Direct-v0 \
    --num_envs 24 --device "$DEV" --enable_cameras \
    --target_offset_xy 1 1 \
    --track \
    --wandb-project-name fish_articulation_analytic_water \
    --wandb-entity ANON-ENTITY \
    --wandb-name salmon_swim_run \
    "$@"
