#!/usr/bin/env bash
set -euo pipefail

# Generate a horizontal grid from a manifest sample.
# Example:
#   GPU_IDS=1 MV_CKPT=outputs/flux_mvs_stage1_v2/mv_adapter_last.pt bash scripts/30_infer_flux_mvs_manifest.sh

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

python src/infer_flux_multiview.py \
  --model_name "${MODEL_NAME}" \
  --mv_ckpt "${MV_CKPT:-outputs/flux_mvs_stage1_v2/mv_adapter_last.pt}" \
  --manifest "${MANIFEST:-data/val_samples.jsonl}" \
  --sample_index "${SAMPLE_INDEX:-0}" \
  --num_views "${NUM_VIEWS:-4}" \
  --height "${HEIGHT:-512}" \
  --width "${WIDTH:-512}" \
  --num_steps "${NUM_STEPS:-30}" \
  --guidance "${GUIDANCE:-3.5}" \
  --seed "${SEED:-42}" \
  --mv_adapter_dim "${MV_ADAPTER_DIM:-512}" \
  --out "${OUT:-outputs/flux_mv_demo.jpg}" \
  ${SINGLE_FLAG} \
  ${HF_FLAG}
