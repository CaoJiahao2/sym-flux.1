#!/usr/bin/env bash
set -euo pipefail

# Run from flux_syncam project root.
# Example: GPU_IDS=1 bash scripts/11_check_flux_mv_forward.sh

source scripts/00_local_flux_env.sh
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

HF_FLAG=""
if [[ "${HF_DOWNLOAD}" == "1" ]]; then
  HF_FLAG="--hf_download"
fi

python src/check_flux_mv_forward.py \
  --model_name "${MODEL_NAME}" \
  --num_views "${NUM_VIEWS:-2}" \
  --mv_adapter_dim "${MV_ADAPTER_DIM:-128}" \
  ${HF_FLAG}
