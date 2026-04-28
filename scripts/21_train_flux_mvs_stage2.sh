#!/usr/bin/env bash
set -euo pipefail

# Stage 2: 4 views.
# Example:
#   GPU_IDS=1 RESUME_MV_CKPT=outputs/flux_mvs_stage1_v2/mv_adapter_last.pt bash scripts/21_train_flux_mvs_stage2.sh

source scripts/00_local_flux_env.sh
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

HF_FLAG=""
if [[ "${HF_DOWNLOAD}" == "1" ]]; then
  HF_FLAG="--hf_download"
fi

SINGLE_FLAG=""
if [[ "${INJECT_SINGLE_BLOCKS:-0}" == "1" ]]; then
  SINGLE_FLAG="--inject_single_blocks"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_MV_CKPT:-}" ]]; then
  RESUME_ARGS+=(--resume_mv_ckpt "${RESUME_MV_CKPT}")
fi

python src/train_flux_multiview.py \
  --model_name "${MODEL_NAME}" \
  --train_manifest "${TRAIN_MANIFEST:-data/train_samples.jsonl}" \
  --output_dir "${OUTPUT_DIR:-outputs/flux_mvs_stage2_v4}" \
  --resolution "${RESOLUTION:-512}" \
  --num_views "${NUM_VIEWS:-4}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --grad_accum "${GRAD_ACCUM:-8}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --max_steps "${MAX_STEPS:-20000}" \
  --learning_rate "${LR:-1e-4}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --guidance "${GUIDANCE:-3.5}" \
  --mv_adapter_dim "${MV_ADAPTER_DIM:-512}" \
  --noise_share_ratio "${NOISE_SHARE_RATIO:-0.75}" \
  --gradient_checkpointing \
  --save_every "${SAVE_EVERY:-1000}" \
  "${RESUME_ARGS[@]}" \
  ${SINGLE_FLAG} \
  ${HF_FLAG}
