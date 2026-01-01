#!/usr/bin/env bash
# Retrain the implicit flow refiner with INFERENCE-CONSISTENT coarse: the coarse
# input is now the 2-view silhouette CARVE of each GT mesh (unit-cube frame,
# rectangular cross-sections, no roundness) instead of degraded-GT. This matches
# what we feed at inference (carve of a real fish). GT SDF + visual (black-bg
# swim video) are unchanged, so we REUSE the v3 SDF cache (no precompute).
#
#   coarse   = Dataset/Fish2VoxelMergedV2CarveCoarse/<asset>/{qwen_voxel64,coarse64_ag*}.npy
#   mix      = 100% "qwen" slot (== carve), tilt-augmented during training
set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-/work/yzha1/miniforge3/envs/simfishlib}"
export PATH="$CONDA_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# DataLoader workers' multiprocessing temp dir on NFS leaves ".nfs* busy" errors
# at teardown (benign, but they make the training process exit 1 and would abort
# the script before eval). Route temp to node-local RAM to avoid it.
export TMPDIR="${TMPDIR:-/dev/shm/$USER-flowtmp}"
mkdir -p "$TMPDIR"

OUT="outputs/experimental/flow_transformer_refiner_implicit_carve_v2_edge"
# reuse the v3 thickened SDF cache (identical mesh/thicken/resolution -> same hash)
CACHE_ROOT="outputs/experimental/flow_transformer_refiner_implicit_thicken_v3/cache_implicit_awrap"
CARVE_ROOT="Dataset/Fish2VoxelMergedV2CarveCoarse"
mkdir -p "$OUT"

FIN_THK=0.02
THIN_THR=0.04
NPROC="${NPROC:-2}"
EPOCHS="${EPOCHS:-400}"
# NOTE: --carve-augment is 0 below. The tilt-augmented variants were frozen
# per-asset across epochs (index-only RNG, no per-epoch reseed) and misregistered
# (rotated-mesh hull vs upright SDF target), which would re-introduce the very
# train/inference mismatch this run removes. Base carve = containment 1.0 and is
# byte-identical in frame to carve_voxel_from_two_views.py inference. Proper
# augmentation (per-epoch reseed + registered small tilts) is a follow-up.

echo "[run] training carve-coarse refiner ($NPROC GPU, $EPOCHS epochs), cache reuse: $CACHE_ROOT"
set +e  # tolerate benign NFS worker-teardown exit code; we gate on checkpoint below
python -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29617}" \
  scripts/experimental/train_flow_refiner_implicit.py \
  --output-dir "$OUT" \
  --cache-root "$CACHE_ROOT" \
  --qwen-coarse-root "$CARVE_ROOT" \
  --coarse-source-mix 1.0,0.0,0.0 \
  --carve-augment 0 \
  --epochs "$EPOCHS" \
  --batch-size 2 \
  --num-workers 2 \
  --save-every 20 \
  --log-every 20 \
  --visual-mode video \
  --num-video-frames 4 \
  --lr 1e-4 \
  --warmup-epochs 10 \
  --cosine-schedule \
  --min-lr 1e-6 \
  --hidden-dim 768 --depth 12 --num-heads 12 \
  --feature-dim 128 \
  --decoder-hidden 256 --decoder-layers 5 --decoder-num-freqs 10 \
  --num-query-points 4096 \
  --surface-loss-weight 6.0 \
  --surface-loss-mode exp \
  --surface-loss-tau 0.12 \
  --sdf-recon-weight 0.25 \
  --sdf-recon-mode l1 \
  --eikonal-weight 0.01 \
  --eikonal-band 0.3 \
  --num-eikonal-points 512 \
  --tv-weight 0.001 \
  --num-tv-points 256 \
  --thicken-fins --fin-thickness "$FIN_THK" --thin-threshold "$THIN_THR" \
  --dit-xt-channels 0 \
  --truncation 4.0 \
  --coarse-resolution 64 \
  2>&1 | tee "$OUT/train.log"
set -e
if [ ! -f "$OUT/checkpoint_best.pt" ]; then
  echo "[run] ERROR: training produced no checkpoint_best.pt"; exit 1
fi
echo "[run] training done -> $OUT/checkpoint_best.pt"

echo "[run] eval on carve coarse (real eval, not degraded-GT illusion)"
EVAL_OUT="$OUT/eval_q128_objs"
mkdir -p "$EVAL_OUT"
python scripts/experimental/evaluate_flow_refiner_implicit.py \
  --checkpoint "$OUT/checkpoint_best.pt" \
  --output-dir "$EVAL_OUT" \
  --qwen-coarse-root "$CARVE_ROOT" \
  --query-resolution 128 \
  --num-steps 200 \
  --num-seeds "${EVAL_NUM_SEEDS:-4}" \
  --hidden-dim 768 --depth 12 --num-heads 12 \
  --feature-dim 128 \
  --decoder-hidden 256 --decoder-layers 5 --decoder-num-freqs 10 \
  --visual-mode video \
  --coarse-resolution 64 \
  --dit-xt-channels 0 \
  --thicken-fins --fin-thickness "$FIN_THK" --thin-threshold "$THIN_THR" \
  2>&1 | tee "$EVAL_OUT/eval.log"

echo "[run] done -> $EVAL_OUT/<asset>/final_mesh.obj  summary: $EVAL_OUT/summary.json"
