#!/usr/bin/env bash
set -euo pipefail

# Full-view MVS training stage: angle=[0,30]
# 默认配置：
#   mv_attn_mode=full_view
#   inject_single_blocks=True
#   pseudo_general_prob=0.25
#   noise: train/inference both independent Gaussian noise
#
# 可切换架构：
#   MV_ARCH=adapter     bash scripts/full_view/20_train_full_view_0_30.sh
#   MV_ARCH=full_hidden bash scripts/full_view/20_train_full_view_0_30.sh
#
# full_hidden 会使用 hidden_size=3072 的 FLUX SelfAttention，并从 double_block.img_attn 初始化；
# 显存消耗明显高于 adapter。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export NUM_VIEWS="${NUM_VIEWS:-4}"
export MV_ATTN_MODE="${MV_ATTN_MODE:-full_view}"
export INJECT_SINGLE_BLOCKS="${INJECT_SINGLE_BLOCKS:-1}"
export PSEUDO_GENERAL_PROB="${PSEUDO_GENERAL_PROB:-0.25}"
export PSEUDO_GENERAL_RANDOM_VIEW="${PSEUDO_GENERAL_RANDOM_VIEW:-1}"
export SINGLE_BLOCK_STRIDE="${SINGLE_BLOCK_STRIDE:-4}"
export MV_ARCH="${MV_ARCH:-adapter}"
export MV_ADAPTER_DIM="${MV_ADAPTER_DIM:-512}"

export TRAIN_MANIFEST="${TRAIN_MANIFEST:-data/samples/stride_${FRAME_STRIDE:-8}_angle_0-30_v${NUM_VIEWS}_train_samples.jsonl}"
export INFER_MANIFEST="${INFER_MANIFEST:-data/samples/stride_${FRAME_STRIDE:-8}_angle_0-30_v${NUM_VIEWS}_val_samples.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/full_view_angle_0-30_${MV_ARCH}}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export RESOLUTION="${RESOLUTION:-512}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export LR="${LR:-1e-4}"
export INFER_SAMPLE_INDEX="${INFER_SAMPLE_INDEX:-0}"
export INFER_NUM_STEPS="${INFER_NUM_STEPS:-30}"
export INFER_SEED="${INFER_SEED:-42}"
export INFER_OUT="${INFER_OUT:-${OUTPUT_DIR}/final_inference_angle_0-30.jpg}"

# 第一阶段默认不自动 resume。如需继续训练，手动设置 RESUME_MV_CKPT=/path/to/mv_adapter_last.pt。

echo "[TRAIN] angle=[0,30]"
echo "[TRAIN] TRAIN_MANIFEST=${TRAIN_MANIFEST}"
echo "[TRAIN] INFER_MANIFEST=${INFER_MANIFEST}"
echo "[TRAIN] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[TRAIN] MV_ARCH=${MV_ARCH} MV_ATTN_MODE=${MV_ATTN_MODE} INJECT_SINGLE_BLOCKS=${INJECT_SINGLE_BLOCKS}"
echo "[TRAIN] PSEUDO_GENERAL_PROB=${PSEUDO_GENERAL_PROB}"

bash scripts/20_train_flux_mvs_stage1.sh
