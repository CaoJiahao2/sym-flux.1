#!/usr/bin/env bash
set -euo pipefail

# Run from project root: bash scripts/01_build_manifest.sh
# This script only reads SynCamVideo-Dataset and writes jsonl manifests to data/.

DATASET_ROOT="${DATASET_ROOT:-data/SynCamVideo-Dataset}"
APERTURE="${APERTURE:-f24_aperture5}"
FRAME_STRIDE="${FRAME_STRIDE:-8}"
NUM_VIEWS="${NUM_VIEWS:-0}"          # 0 means keep cam01-cam10 if available
MAX_ANGLE="${MAX_ANGLE:-}"           # empty means no angle filter
MAX_TRAIN_SCENES="${MAX_TRAIN_SCENES:-0}"  # 0 means all scenes
MAX_VAL_SCENES="${MAX_VAL_SCENES:-0}"      # 0 means all scenes
SAMPLING="${SAMPLING:-first}"
SEED="${SEED:-1234}"
OUT_BASE="data/samples/stride_${FRAME_STRIDE}_angle_${MAX_ANGLE:-all}_v${NUM_VIEWS}"

mkdir -p data outputs

# 修复2：使用独立的变量避免覆盖污染
CAPTIONS_TRAIN="${CAPTIONS_TRAIN:-data/captions/captions_scene_level_train.json}"
if [[ ! -f "${CAPTIONS_TRAIN}" ]]; then
  echo "[WARN] ${CAPTIONS_TRAIN} not found. Creating an empty captions.json; fallback prompt will be used."
  mkdir -p "$(dirname "${CAPTIONS_TRAIN}")" # 修复3：确保目录存在
  printf '{}\n' > "${CAPTIONS_TRAIN}"
fi

CAPTIONS_VAL="${CAPTIONS_VAL:-data/captions/captions_scene_level_val.json}"
if [[ ! -f "${CAPTIONS_VAL}" ]]; then
  echo "[WARN] ${CAPTIONS_VAL} not found. Creating an empty captions.json; fallback prompt will be used."
  mkdir -p "$(dirname "${CAPTIONS_VAL}")"   # 修复3：确保目录存在
  printf '{}\n' > "${CAPTIONS_VAL}"
fi

COMMON_ARGS=(
  --dataset_root "${DATASET_ROOT}"
  --aperture "${APERTURE}"
  --frame_stride "${FRAME_STRIDE}"
  --num_views "${NUM_VIEWS}"
  --sampling "${SAMPLING}"
  --seed "${SEED}"
)

if [[ -n "${MAX_ANGLE}" ]]; then
  COMMON_ARGS+=(--max_angle "${MAX_ANGLE}")
fi

python src/data/build_manifest.py \
  "${COMMON_ARGS[@]}" \
  --split train \
  --max_scenes "${MAX_TRAIN_SCENES}" \
  --captions "${CAPTIONS_TRAIN}" \
  --out "${OUT_BASE}_train_samples.jsonl"

python src/data/build_manifest.py \
  "${COMMON_ARGS[@]}" \
  --split val \
  --max_scenes "${MAX_VAL_SCENES}" \
  --captions "${CAPTIONS_VAL}" \
  --out "${OUT_BASE}_val_samples.jsonl"

# 修复1：使用正确的变量进行校验，防止 set -e 导致脚本中断
wc -l "${OUT_BASE}_train_samples.jsonl" "${OUT_BASE}_val_samples.jsonl"
head -n 1 "${OUT_BASE}_train_samples.jsonl" | python -m json.tool | head -n 20