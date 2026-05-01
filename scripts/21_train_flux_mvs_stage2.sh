#!/usr/bin/env bash
set -euo pipefail

# Stage 2: more views or larger view gap.
# For closer SynCamMaster behavior, try MV_ATTN_MODE=full_view after same_token is stable.
# Enable selected single-stream adapters with INJECT_SINGLE_BLOCKS=1 SINGLE_BLOCK_STRIDE=4.

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

NO_MV_MOD_FLAG=""
if [[ "${NO_MV_TIMESTEP_MODULATION:-0}" == "1" ]]; then
  NO_MV_MOD_FLAG="--no_mv_timestep_modulation"
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
  --mv_attn_mode "${MV_ATTN_MODE:-same_token}" \
  --single_block_stride "${SINGLE_BLOCK_STRIDE:-4}" \
  # 训练时过度共享噪声，可能会让模型学成“多张图尽量一样”，削弱视角变化,noise_share_ratio = 0.0 或很小
  --noise_share_ratio "${NOISE_SHARE_RATIO:-0.0}" \ 
  --gradient_checkpointing \
  --save_every "${SAVE_EVERY:-1000}" \
  "${RESUME_ARGS[@]}" \
  ${SINGLE_FLAG} \
  ${NO_MV_MOD_FLAG} \
  ${HF_FLAG}
