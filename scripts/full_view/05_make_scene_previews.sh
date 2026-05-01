#!/usr/bin/env bash
set -euo pipefail

# 可选：为场景级 caption 制作预览拼图。
# 默认同时处理 train/val。
#
# 使用：
#   DATASET_ROOT=data/SynCamVideo-Dataset bash scripts/full_view/05_make_scene_previews.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET_ROOT="${DATASET_ROOT:-data/SynCamVideo-Dataset}"
THUMB_SIZE="${THUMB_SIZE:-256}"

python src/caption/make_scene_previews.py \
  --dataset_root "${DATASET_ROOT}" \
  --split train \
  --out_dir data/scene_previews/train \
  --thumb_size "${THUMB_SIZE}"

python src/caption/make_scene_previews.py \
  --dataset_root "${DATASET_ROOT}" \
  --split val \
  --out_dir data/scene_previews/val \
  --thumb_size "${THUMB_SIZE}"
