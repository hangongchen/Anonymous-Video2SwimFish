#!/usr/bin/env bash
# Hierarchical Qwen + flow-implicit refiner inference over Dataset/testset.
# Uses the no-Eikonal checkpoint (the better of the two implicit runs).
#
# Per input image, writes <OUTPUT_ROOT>/<slug>/:
#   8.obj  / 8.png        - Qwen 8^3 coarse, upsampled to boxy 64^3
#   64.obj / 64.png       - Qwen 64^3 voxel
#   mesh.obj / mesh.png   - flow-implicit refiner (marching cubes on TSDF)
#   log/qwen/             - raw Qwen outputs
#   log/flow/             - sdf.npy + summary.json
#   log/input.<ext>       - copy of the source image
#
# Qwen32B fits on one H200. Run from a single-GPU allocation, inside tmux:
#   srun --pty --gres=gpu:h200:1 --cpus-per-task=32 --mem=256G --time=4:00:00 bash
#   tmux new -s testset_flow
#   bash scripts/infer_testset_flow_implicit.sh
set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-/work/yzha1/miniforge3/envs/simfishlib}"
if [[ ! -x "$CONDA_ENV/bin/python" ]]; then
  echo "[infer_testset_flow] $CONDA_ENV/bin/python not found" >&2
  exit 1
fi
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOCAL_TMP_BASE="${LOCAL_TMP_BASE:-/dev/shm/${USER}}"
mkdir -p "$LOCAL_TMP_BASE"
export TMPDIR="${TMPDIR:-$LOCAL_TMP_BASE}"

TESTSET_DIR="${TESTSET_DIR:-Dataset/testset}"
CKPT="${CKPT:-outputs/experimental/flow_transformer_refiner_implicit/checkpoint_best.pt}"
QWEN_LAUNCHER="${QWEN_LAUNCHER:-scripts/reproduce_checkpoint3350_infer.sh}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/experimental/flow_transformer_refiner_implicit/testset_inference}"
QUERY_RES="${QUERY_RES:-128}"
NUM_STEPS="${NUM_STEPS:-200}"
BATCH_QUERY="${BATCH_QUERY:-65536}"

export REFINE_BATCH_SIZE="${REFINE_BATCH_SIZE:-12}"
export QWEN3VL_DEVICE_MAP="${QWEN3VL_DEVICE_MAP:-none}"

EXTRA=()
if [[ "${SKIP_QWEN_IF_CACHED:-1}" == "1" ]]; then
  EXTRA+=(--skip-qwen-if-cached)
fi

mkdir -p "$OUTPUT_ROOT"
echo "TESTSET_DIR     = $TESTSET_DIR"
echo "OUTPUT_ROOT     = $OUTPUT_ROOT"
echo "CKPT            = $CKPT"
echo "QWEN_LAUNCHER   = $QWEN_LAUNCHER  (refine_batch_size=$REFINE_BATCH_SIZE)"
echo "QUERY_RES       = $QUERY_RES  (num_steps=$NUM_STEPS)"
echo "SKIP_QWEN_CACHE = ${SKIP_QWEN_IF_CACHED:-1}"
echo ""

python scripts/experimental/infer_testset_flow_implicit.py \
  --testset-dir   "$TESTSET_DIR" \
  --output-root   "$OUTPUT_ROOT" \
  --checkpoint    "$CKPT" \
  --qwen-launcher "$QWEN_LAUNCHER" \
  --query-resolution "$QUERY_RES" \
  --num-steps      "$NUM_STEPS" \
  --batch-query    "$BATCH_QUERY" \
  "${EXTRA[@]}"

echo ""
echo "Done. Per-image outputs under $OUTPUT_ROOT/"
