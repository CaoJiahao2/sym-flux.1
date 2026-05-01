#!/usr/bin/env bash
set -euo pipefail

# 功能检查：只测试 FLUX-MVS 前向传播，不读数据集，不训练。
# 默认：full_view + inject_single_blocks=True + pseudo_general_prob=0.25。
# 如需测试 full hidden 版本：
#   MV_ARCH=full_hidden bash scripts/full_view/00_check_forward_full_view.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source scripts/00_local_flux_env.sh
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

HF_FLAG=""
if [[ "${HF_DOWNLOAD:-0}" == "1" ]]; then
  HF_FLAG="--hf_download"
fi

INJECT_FLAG="--inject_single_blocks"
if [[ "${INJECT_SINGLE_BLOCKS:-1}" == "0" ]]; then
  INJECT_FLAG="--no_inject_single_blocks"
fi

python src/check_flux_mv_forward.py \
  --model_name "${MODEL_NAME:-flux-dev}" \
  --num_views "${NUM_VIEWS:-4}" \
  --seq_len "${SEQ_LEN:-1024}" \
  --txt_len "${TXT_LEN:-16}" \
  --mv_arch "${MV_ARCH:-adapter}" \
  --mv_adapter_dim "${MV_ADAPTER_DIM:-512}" \
  --mv_attn_mode "${MV_ATTN_MODE:-full_view}" \
  --single_block_stride "${SINGLE_BLOCK_STRIDE:-4}" \
  ${INJECT_FLAG} \
  ${HF_FLAG}
