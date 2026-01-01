#!/usr/bin/env bash
# Single-GPU launcher for Template-Salmon-Swim-Direct-v0 WITH the FEM deformable body
# ACTIVE (env.with_deformable=true). The soft body adds real attachment/inertia/damping
# dynamics on the skeleton, so a policy must be trained with it (a skeleton-only policy
# does NOT transfer -- different effective inertia, viscoelastic damping, attachment
# reaction forces). Heavier than skeleton-only but fits easily (256 envs ~ 8 GB).
#
# Multi-GPU is unavailable on this box (NCCL "illegal memory access" in the cross-GPU
# broadcast, independent of env count), so this is single-GPU.
#
# Usage:
#   bash scripts/train_salmon_swim_soft.sh                  # cuda:0, 256 envs, with soft body
#   DEV=cuda:1 bash scripts/train_salmon_swim_soft.sh ...   # other GPU / 2nd independent run
#   bash scripts/train_salmon_swim_soft.sh --num_envs 512   # scale up (GPU has headroom)
set -u

REPO=${FISH_ROOT}
PY=${FISH_PYTHON}
DEV=${DEV:-cuda:0}
# HEADLESS=1 (default): no GUI (fast). HEADLESS=0: open the Isaac Sim GUI viewport to watch the
# fish swim live (slower -- rendering overhead; use a small --num_envs). Needs a display (DISPLAY).
HEADLESS=${HEADLESS:-1}
HEADLESS_FLAG=""
[ "$HEADLESS" = "1" ] && HEADLESS_FLAG="--headless"

echo "[launch] cleaning up leftover salmon-swim training..."
pkill -9 -f "Template-Salmon-Swim-Direct-v0.*--headless" 2>/dev/null || true
pkill -9 -f "train_ppo.py.*--distributed" 2>/dev/null || true
sleep 2

cd "$REPO"
export WANDB_MODE=${WANDB_MODE:-online}
export HYDRA_FULL_ERROR=1
export DISPLAY=${DISPLAY:-:1}

# NOTE: recording is the cfg default but it is DISABLED here (env.record_video=false) and
# no --enable_cameras -- the RTX recording camera STALLS on the deformable's Replicator
# shader compile. Record on a skeleton-only run instead.
exec "$PY" scripts/rl_games/train_ppo.py \
    --task Template-Salmon-Swim-Direct-v0 \
    --num_envs 256 --device "$DEV" $HEADLESS_FLAG \
    --target_offset_xy 1 1 \
    --track \
    --wandb-project-name fish_articulation_analytic_water \
    --wandb-entity ANON-ENTITY \
    --wandb-name salmon_swim_soft \
    env.with_deformable=true \
    env.record_video=false \
    "$@"
