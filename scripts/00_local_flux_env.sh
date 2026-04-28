#!/usr/bin/env bash

# Use like: GPU_IDS=7 bash scripts/11_check_flux_mv_forward.sh
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"

# Local diffusers/BFL checkpoint directory.
export LOCAL_FLUX_DIR="${LOCAL_FLUX_DIR:-/data/model_cjh/FLUX.1-dev}"

# Compatible with different versions/forks of black-forest-labs/flux.
export FLUX_DEV="${FLUX_DEV:-${LOCAL_FLUX_DIR}/flux1-dev.safetensors}"
export FLUX_MODEL="${FLUX_MODEL:-${FLUX_DEV}}"
export AE="${AE:-${LOCAL_FLUX_DIR}/ae.safetensors}"
export FLUX_AE="${FLUX_AE:-${AE}}"

export MODEL_NAME="${MODEL_NAME:-flux-dev}"
export HF_DOWNLOAD="${HF_DOWNLOAD:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -f "${FLUX_DEV}" ]]; then
  echo "[ERROR] FLUX_DEV not found: ${FLUX_DEV}" >&2
  exit 1
fi
if [[ ! -f "${AE}" ]]; then
  echo "[ERROR] AE not found: ${AE}" >&2
  exit 1
fi

echo "[ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[ENV] MODEL_NAME=${MODEL_NAME}"
echo "[ENV] LOCAL_FLUX_DIR=${LOCAL_FLUX_DIR}"
echo "[ENV] FLUX_DEV=${FLUX_DEV}"
echo "[ENV] FLUX_MODEL=${FLUX_MODEL}"
echo "[ENV] AE=${AE}"
echo "[ENV] FLUX_AE=${FLUX_AE}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
