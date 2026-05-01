#!/usr/bin/env bash
set -euo pipefail

# 一次性构建三个角度区间的数据清单：
#   [0,30], [15,45], [30,60]
#
# 默认输出：
#   data/samples/stride_8_angle_0-30_v4_train_samples.jsonl
#   data/samples/stride_8_angle_0-30_v4_val_samples.jsonl
#   data/samples/stride_8_angle_15-45_v4_train_samples.jsonl
#   data/samples/stride_8_angle_15-45_v4_val_samples.jsonl
#   data/samples/stride_8_angle_30-60_v4_train_samples.jsonl
#   data/samples/stride_8_angle_30-60_v4_val_samples.jsonl

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export DATASET_ROOT="${DATASET_ROOT:-data/SynCamVideo-Dataset}"
export APERTURE="${APERTURE:-f24_aperture5}"
export FRAME_STRIDE="${FRAME_STRIDE:-8}"
export NUM_VIEWS="${NUM_VIEWS:-4}"
export SAMPLING="${SAMPLING:-random}"
export SEED="${SEED:-1234}"

# 调试时可以限制场景数量，例如：
#   MAX_TRAIN_SCENES=50 MAX_VAL_SCENES=10 bash scripts/full_view/10_build_manifests_full_view_angles.sh
export MAX_TRAIN_SCENES="${MAX_TRAIN_SCENES:-0}"
export MAX_VAL_SCENES="${MAX_VAL_SCENES:-0}"

build_one() {
  local min_angle="$1"
  local max_angle="$2"
  echo ""
  echo "========== Build manifest: angle=[${min_angle},${max_angle}], V=${NUM_VIEWS} =========="
  MIN_ANGLE="${min_angle}" MAX_ANGLE="${max_angle}" bash scripts/01_build_manifest.sh
}

build_one 0 30
build_one 15 45
build_one 30 60

echo ""
echo "Done. Generated manifests under data/samples/."
