#!/usr/bin/env bash
set -euo pipefail

# Stage 1: 2 views.
# Example:
#   GPU_IDS=1 TRAIN_MANIFEST=data/train_samples.jsonl bash scripts/20_train_flux_mvs_stage1.sh

source scripts/00_local_flux_env.sh
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

HF_FLAG=""
if [[ "${HF_DOWNLOAD}" == "1" ]]; then
  HF_FLAG="--hf_download"
fi

python src/train_flux_multiview.py \
  --model_name "${MODEL_NAME}" \
  --train_manifest "${TRAIN_MANIFEST:-data/train_samples.jsonl}" \
  --output_dir "${OUTPUT_DIR:-outputs/flux_mvs_stage1_v2}" \
  --resolution "${RESOLUTION:-512}" \
  --num_views "${NUM_VIEWS:-2}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --grad_accum "${GRAD_ACCUM:-8}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --max_steps "${MAX_STEPS:-5000}" \
  --learning_rate "${LR:-1e-4}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --guidance "${GUIDANCE:-3.5}" \
  --mv_attn_mode full_view \
  --mv_adapter_dim "${MV_ADAPTER_DIM:-512}" \
  --noise_share_ratio "${NOISE_SHARE_RATIO:-0.75}" \
  --gradient_checkpointing \
  --save_every "${SAVE_EVERY:-500}" \
  ${HF_FLAG}
