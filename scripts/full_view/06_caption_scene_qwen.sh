#!/usr/bin/env bash
set -euo pipefail

# 可选：用本地 Qwen/VLM 对 scene preview 生成场景级 caption。
# 注意 caption 不应包含 view/camera/left/right/front/back 等视角词。
#
# 使用：
#   QWEN_MODEL_PATH=/path/to/Qwen3.5-9B GPU_ID=0 bash scripts/full_view/06_caption_scene_qwen.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/data/model_cjh/Qwen/Qwen3.5-9B}"
GPU_ID="${GPU_ID:-0}"
MAX_WORDS="${MAX_WORDS:-50}"

mkdir -p data/captions

python src/caption/caption_scene_level_qwen35.py \
  --model_path "${QWEN_MODEL_PATH}" \
  --preview_dir data/scene_previews/train \
  --out data/captions/captions_scene_level_train.json \
  --max_words "${MAX_WORDS}" \
  --gpu_id "${GPU_ID}" \
  --resume

python src/caption/check_captions.py \
  --caption_json data/captions/captions_scene_level_train.json \
  --max_words "${MAX_WORDS}"

python src/caption/caption_scene_level_qwen35.py \
  --model_path "${QWEN_MODEL_PATH}" \
  --preview_dir data/scene_previews/val \
  --out data/captions/captions_scene_level_val.json \
  --max_words "${MAX_WORDS}" \
  --gpu_id "${GPU_ID}" \
  --resume

python src/caption/check_captions.py \
  --caption_json data/captions/captions_scene_level_val.json \
  --max_words "${MAX_WORDS}"
