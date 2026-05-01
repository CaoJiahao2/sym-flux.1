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

python src/check_flux_mv_forward.py \
  --model_name "${MODEL_NAME}" \
  --num_views "${NUM_VIEWS:-2}" \
  --mv_arch "${MV_ARCH:-adapter}" \
  --mv_adapter_dim "${MV_ADAPTER_DIM:-128}" \
  --mv_attn_mode "${MV_ATTN_MODE:-full_view}" \
  --single_block_stride "${SINGLE_BLOCK_STRIDE:-4}" \
  ${SINGLE_FLAG} \
  ${NO_MV_MOD_FLAG} \
  ${HF_FLAG}
