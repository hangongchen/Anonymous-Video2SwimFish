#!/usr/bin/env bash
# VLM fish-rigger fine-tune (ms-swift LoRA): overlay renders -> next rigging action.
# Dataset: simfishlib_data/vlm_rigger/sft/rig_sft.jsonl (built by
# simfishlib/experimental/vlm_rigger/build_sft_jsonl.py). Two images per row
# (side + top view), short JSON action targets, long shared system prompt.
set -euo pipefail
cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-/work/yzha1/miniforge3/envs/simfishlib}"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"

# See train_qwen3vl_part64_ms_swift.sh: stub CUDA_HOME so accelerate's deepspeed
# import survives without nvcc.
export CUDA_HOME="${CUDA_HOME:-$(pwd)/.fake_cuda}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NGPU=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export NPROC_PER_NODE="${NPROC_PER_NODE:-$NGPU}"
echo "[train] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  NPROC_PER_NODE=$NPROC_PER_NODE"

MODEL_PATH="${MODEL_PATH:-models/Qwen3-VL-32B-Instruct}"
DATASET_PATH="${DATASET_PATH:-simfishlib_data/vlm_rigger/sft/rig_sft.jsonl}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-simfishlib_data/vlm_rigger/sft/rig_sft_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3vl-fish-rigger}"

EXTRA=()
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max_steps "$MAX_STEPS")

swift sft \
  --model "$MODEL_PATH" \
  --model_type qwen3_vl \
  --dataset "$DATASET_PATH" \
  --val_dataset "$VAL_DATASET_PATH" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --max_length "${MAX_LENGTH:-4096}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-4}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --target_modules all-linear \
  --save_steps "${SAVE_STEPS:-50}" \
  --eval_steps "${EVAL_STEPS:-50}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA[@]}"
