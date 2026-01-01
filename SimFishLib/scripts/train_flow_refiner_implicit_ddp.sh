#!/usr/bin/env bash
# Two-GPU DDP training for the implicit-decoder flow refiner.
# Run inside tmux on a node with 2 H200s:
#
#   tmux new -s flow_implicit
#   bash scripts/train_flow_refiner_implicit_ddp.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-/work/yzha1/miniforge3/envs/simfishlib}"
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MASTER_PORT="${MASTER_PORT:-29582}"

OUTPUT="${OUTPUT:-outputs/experimental/flow_transformer_refiner_implicit}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
HIDDEN_DIM="${HIDDEN_DIM:-768}"
DEPTH="${DEPTH:-12}"
NUM_HEADS="${NUM_HEADS:-12}"
FEATURE_DIM="${FEATURE_DIM:-128}"
DECODER_HIDDEN="${DECODER_HIDDEN:-256}"
DECODER_LAYERS="${DECODER_LAYERS:-5}"
DECODER_NUM_FREQS="${DECODER_NUM_FREQS:-10}"
NUM_QUERY_POINTS="${NUM_QUERY_POINTS:-4096}"
SURFACE_LOSS_WEIGHT="${SURFACE_LOSS_WEIGHT:-4.0}"
CACHE_ROOT="${CACHE_ROOT:-$OUTPUT/cache_implicit}"

mkdir -p "$OUTPUT"

python -m torch.distributed.run --nproc_per_node=2 --master_port="$MASTER_PORT" \
  scripts/experimental/train_flow_refiner_implicit.py \
  --output-dir "$OUTPUT" \
  --cache-root "$CACHE_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers 2 \
  --save-every 20 \
  --log-every 20 \
  --visual-mode video \
  --num-video-frames 4 \
  --lr 2e-4 \
  --hidden-dim "$HIDDEN_DIM" --depth "$DEPTH" --num-heads "$NUM_HEADS" \
  --feature-dim "$FEATURE_DIM" \
  --decoder-hidden "$DECODER_HIDDEN" \
  --decoder-layers "$DECODER_LAYERS" \
  --decoder-num-freqs "$DECODER_NUM_FREQS" \
  --num-query-points "$NUM_QUERY_POINTS" \
  --surface-loss-weight "$SURFACE_LOSS_WEIGHT" \
  2>&1 | tee "$OUTPUT/train.log"
