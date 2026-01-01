#!/usr/bin/env bash
# Adds the four-loss recipe on top of run_flow_implicit_thicken_v2.sh:
#   loss = flow_mse + 0.25 * sdf_recon_l1 + 0.01 * eikonal + 0.001 * tv
# plus exponential surface weighting (3x at the surface, tau=0.03).
#
# All four are computed via the existing finite-diff machinery, so no double-
# backward issues. Eikonal/TV add ~4*512 + 4*256 extra decoder queries/step.
set -eo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-/work/yzha1/miniforge3/envs/simfishlib}"
export PATH="$CONDA_ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT="outputs/experimental/flow_transformer_refiner_implicit_thicken_v3"
CACHE_ROOT="$OUT/cache_implicit_awrap"
mkdir -p "$OUT"

# CGAL alpha-wrap solidification (global; fin thickness ~= FIN_THK) + 4-way
# near-surface-heavy point sampling. Thinner fins than before (0.04 -> 0.02).
FIN_THK=0.02
THIN_THR=0.04

# Cache hash bakes in fin params + the alpha-wrap/sampling version, so any stale
# cache from an earlier thickening/sampling pass won't be reused.
if ! ls "$CACHE_ROOT" 2>/dev/null | grep -q implicit_v4_; then
  echo "[run] precomputing cache (thicken=ON, fin-thickness=$FIN_THK, thin-threshold=$THIN_THR)"
  python scripts/experimental/precompute_flow_implicit.py \
    --output-dir "$OUT" --cache-root "$CACHE_ROOT" \
    --thicken-fins --fin-thickness "$FIN_THK" --thin-threshold "$THIN_THR" \
    --coarse-resolution 64 --truncation 4.0
else
  echo "[run] reusing thickened cache at $CACHE_ROOT"
fi

echo "[run] training (4-loss recipe, lr 1e-4 cosine, warmup 10, 400 epochs, 2 GPUs)"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29589 \
  scripts/experimental/train_flow_refiner_implicit.py \
  --output-dir "$OUT" \
  --cache-root "$CACHE_ROOT" \
  --epochs 400 \
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
  --surface-loss-weight 3.0 \
  --surface-loss-mode exp \
  --surface-loss-tau 0.3 \
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

echo "[run] eval (query resolution 128, 200 Euler steps, ${EVAL_NUM_SEEDS:-4}-seed avg)"
EVAL_OUT="$OUT/eval_q128_objs"
mkdir -p "$EVAL_OUT"
python scripts/experimental/evaluate_flow_refiner_implicit.py \
  --checkpoint "$OUT/checkpoint_best.pt" \
  --output-dir "$EVAL_OUT" \
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

echo "[run] done. predictions: $EVAL_OUT/<asset>/final_mesh.obj   summary: $EVAL_OUT/summary.json"
