#!/usr/bin/env bash
set -euo pipefail

# Run from project root: bash scripts/01_build_manifest.sh
# This script only reads SynCamVideo-Dataset and writes jsonl manifests to data/.

DATASET_ROOT="${DATASET_ROOT:-data/SynCamVideo-Dataset}"
CAPTIONS="${CAPTIONS:-data/captions.json}"
APERTURE="${APERTURE:-f24_aperture5}"
FRAME_STRIDE="${FRAME_STRIDE:-8}"
CONVENTION="${CONVENTION:-w2c}"
NUM_VIEWS="${NUM_VIEWS:-0}"          # 0 means keep cam01-cam10 if available
MAX_ANGLE="${MAX_ANGLE:-}"           # empty means no angle filter
MAX_TRAIN_SCENES="${MAX_TRAIN_SCENES:-0}"  # 0 means all scenes
MAX_VAL_SCENES="${MAX_VAL_SCENES:-0}"      # 0 means all scenes
SAMPLING="${SAMPLING:-first}"
SEED="${SEED:-1234}"

mkdir -p data outputs

if [[ ! -f "${CAPTIONS}" ]]; then
  echo "[WARN] ${CAPTIONS} not found. Creating an empty captions.json; fallback prompt will be used."
  printf '{}\n' > "${CAPTIONS}"
fi

COMMON_ARGS=(
  --dataset_root "${DATASET_ROOT}"
  --aperture "${APERTURE}"
  --captions "${CAPTIONS}"
  --frame_stride "${FRAME_STRIDE}"
  --num_views "${NUM_VIEWS}"
  --sampling "${SAMPLING}"
  --seed "${SEED}"
  --convention "${CONVENTION}"
)

if [[ -n "${MAX_ANGLE}" ]]; then
  COMMON_ARGS+=(--max_angle "${MAX_ANGLE}")
fi

python src/data/build_manifest.py \
  "${COMMON_ARGS[@]}" \
  --split train \
  --max_scenes "${MAX_TRAIN_SCENES}" \
  --out data/train_samples.jsonl

python src/data/build_manifest.py \
  "${COMMON_ARGS[@]}" \
  --split val \
  --max_scenes "${MAX_VAL_SCENES}" \
  --out data/val_samples.jsonl

wc -l data/train_samples.jsonl data/val_samples.jsonl
head -n 1 data/train_samples.jsonl | python -m json.tool | head -n 80
