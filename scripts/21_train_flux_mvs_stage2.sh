#!/usr/bin/env bash
set -euo pipefail

# Stage 2: more views or larger view gap. Defaults keep full_view attention.
# Use MV_ARCH=full_hidden for the hidden-size variant copied from FLUX img_attn.

source scripts/00_local_flux_env.sh
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

HF_FLAG=""
if [[ "${HF_DOWNLOAD:-0}" == "1" ]]; then
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

PSEUDO_RANDOM_FLAG=""
if [[ "${PSEUDO_GENERAL_RANDOM_VIEW:-0}" == "1" ]]; then
  PSEUDO_RANDOM_FLAG="--pseudo_general_random_view"
fi

INFER_AFTER_FLAG=""
if [[ "${NO_INFER_AFTER_TRAINING:-0}" == "1" ]]; then
  INFER_AFTER_FLAG="--no_infer_after_training"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_MV_CKPT:-}" ]]; then
  RESUME_ARGS+=(--resume_mv_ckpt "${RESUME_MV_CKPT}")
fi

INFER_ARGS=()
if [[ -n "${INFER_MANIFEST:-}" ]]; then
  INFER_ARGS+=(--infer_manifest "${INFER_MANIFEST}")
fi
if [[ -n "${INFER_OUT:-}" ]]; then
  INFER_ARGS+=(--infer_out "${INFER_OUT}")
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
  --mv_arch "${MV_ARCH:-adapter}" \
  --mv_adapter_dim "${MV_ADAPTER_DIM:-512}" \
  --mv_attn_mode "${MV_ATTN_MODE:-full_view}" \
  --single_block_stride "${SINGLE_BLOCK_STRIDE:-4}" \
  --pseudo_general_prob "${PSEUDO_GENERAL_PROB:-0.15}" \
  --infer_sample_index "${INFER_SAMPLE_INDEX:-0}" \
  --infer_num_steps "${INFER_NUM_STEPS:-30}" \
  --infer_seed "${INFER_SEED:-42}" \
  --gradient_checkpointing \
  --save_every "${SAVE_EVERY:-1000}" \
  "${RESUME_ARGS[@]}" \
  "${INFER_ARGS[@]}" \
  ${SINGLE_FLAG} \
  ${NO_MV_MOD_FLAG} \
  ${PSEUDO_RANDOM_FLAG} \
  ${INFER_AFTER_FLAG} \
  ${HF_FLAG}
