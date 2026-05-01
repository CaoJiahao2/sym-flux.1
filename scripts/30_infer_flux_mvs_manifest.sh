#!/usr/bin/env bash
set -euo pipefail

source scripts/00_local_flux_env.sh
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

HF_FLAG=""
if [[ "${HF_DOWNLOAD:-0}" == "1" ]]; then
  HF_FLAG="--hf_download"
fi

SINGLE_FLAG="--inject_single_blocks"
if [[ "${INJECT_SINGLE_BLOCKS:-1}" == "0" ]]; then
  SINGLE_FLAG="--no_inject_single_blocks"
fi

NO_MV_MOD_FLAG=""
if [[ "${NO_MV_TIMESTEP_MODULATION:-0}" == "1" ]]; then
  NO_MV_MOD_FLAG="--no_mv_timestep_modulation"
fi

python src/infer_flux_multiview.py \
  --model_name "${MODEL_NAME}" \
  --mv_ckpt "${MV_CKPT}" \
  --manifest "${MANIFEST:-data/val_samples.jsonl}" \
  --sample_index "${SAMPLE_INDEX:-0}" \
  --num_views "${NUM_VIEWS:-4}" \
  --height "${HEIGHT:-512}" \
  --width "${WIDTH:-512}" \
  --num_steps "${NUM_STEPS:-30}" \
  --guidance "${GUIDANCE:-3.5}" \
  --seed "${SEED:-42}" \
  --mv_arch "${MV_ARCH:-adapter}" \
  --mv_adapter_dim "${MV_ADAPTER_DIM:-512}" \
  --mv_attn_mode "${MV_ATTN_MODE:-full_view}" \
  --single_block_stride "${SINGLE_BLOCK_STRIDE:-4}" \
  --out "${OUT:-outputs/flux_mv_demo.jpg}" \
  ${SINGLE_FLAG} \
  ${NO_MV_MOD_FLAG} \
  ${HF_FLAG}
