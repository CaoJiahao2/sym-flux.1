#!/usr/bin/env bash
set -euo pipefail

# Run from project root: bash scripts/00_check_data.sh
# This script only reads SynCamVideo-Dataset and writes preview images to outputs/.

DATASET_ROOT="${DATASET_ROOT:-data/SynCamVideo-Dataset}"
SPLIT="${SPLIT:-train}"
APERTURE="${APERTURE:-f24_aperture5}"
SCENE="${SCENE:-scene10}"
FRAME_IDX="${FRAME_IDX:-80}"
RESIZE="${RESIZE:-256}"
OUT="${OUT:-outputs/preview_${SPLIT}_${APERTURE}_${SCENE}_frame${FRAME_IDX}_cam01-cam10.jpg}"

python src/data/preview_multiview.py \
  --dataset_root "${DATASET_ROOT}" \
  --split "${SPLIT}" \
  --aperture "${APERTURE}" \
  --scene "${SCENE}" \
  --frame_idx "${FRAME_IDX}" \
  --resize "${RESIZE}" \
  --out "${OUT}"
