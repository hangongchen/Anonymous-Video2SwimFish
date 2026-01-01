#!/usr/bin/env bash
# Evaluate the implicit-decoder flow refiner. Default query resolution 128^3
# (~4x more mesh detail than the dense 64^3 head; set QUERY_RES=256 for ~16x,
# bigger jump but slower).
set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-/work/yzha1/miniforge3/envs/simfishlib}"
export PATH="$CONDA_ENV/bin:$PATH"

RUN="${RUN:-outputs/experimental/flow_transformer_refiner_implicit}"
CHECKPOINT="${CHECKPOINT:-$RUN/checkpoint_best.pt}"
QUERY_RES="${QUERY_RES:-128}"
OUTPUT="${OUTPUT:-$RUN/eval_q${QUERY_RES}}"
HIDDEN_DIM="${HIDDEN_DIM:-768}"
DEPTH="${DEPTH:-12}"
NUM_HEADS="${NUM_HEADS:-12}"
FEATURE_DIM="${FEATURE_DIM:-128}"
DECODER_HIDDEN="${DECODER_HIDDEN:-256}"
DECODER_LAYERS="${DECODER_LAYERS:-5}"
DECODER_NUM_FREQS="${DECODER_NUM_FREQS:-10}"
NUM_STEPS="${NUM_STEPS:-200}"

mkdir -p "$OUTPUT"

python scripts/experimental/evaluate_flow_refiner_implicit.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT" \
  --query-resolution "$QUERY_RES" \
  --num-steps "$NUM_STEPS" \
  --hidden-dim "$HIDDEN_DIM" --depth "$DEPTH" --num-heads "$NUM_HEADS" \
  --feature-dim "$FEATURE_DIM" \
  --decoder-hidden "$DECODER_HIDDEN" \
  --decoder-layers "$DECODER_LAYERS" \
  --decoder-num-freqs "$DECODER_NUM_FREQS" \
  2>&1 | tee "$OUTPUT/eval.log"
