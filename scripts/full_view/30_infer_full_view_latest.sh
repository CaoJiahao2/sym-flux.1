#!/usr/bin/env bash
set -euo pipefail

# 使用最终阶段 checkpoint 对 val manifest 做一次独立 noise 的多视角推理。
# 默认读取：
#   outputs/full_view_angle_30-60_adapter/mv_adapter_last.pt
#
# 如果训练的是 full_hidden：
#   MV_ARCH=full_hidden bash scripts/full_view/30_infer_full_view_latest.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export NUM_VIEWS="${NUM_VIEWS:-4}"
export MV_ARCH="${MV_ARCH:-adapter}"
export MV_ATTN_MODE="${MV_ATTN_MODE:-full_view}"
export INJECT_SINGLE_BLOCKS="${INJECT_SINGLE_BLOCKS:-1}"
export PSEUDO_GENERAL_PROB="${PSEUDO_GENERAL_PROB:-0.25}"

export MANIFEST="${MANIFEST:-data/samples/stride_${FRAME_STRIDE:-8}_angle_30-60_v${NUM_VIEWS}_val_samples.jsonl}"
export MV_CKPT="${MV_CKPT:-outputs/full_view_angle_30-60_${MV_ARCH}/mv_adapter_last.pt}"
export OUT="${OUT:-outputs/full_view_angle_30-60_${MV_ARCH}/manual_inference.jpg}"

bash scripts/30_infer_flux_mvs_manifest.sh
