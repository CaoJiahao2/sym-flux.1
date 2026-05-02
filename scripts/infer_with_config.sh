#!/usr/bin/env bash
set -euo pipefail

# Generic inference entry. Usage:
#   bash scripts/infer_with_config.sh configs/val/full_view_stage3_30_60_v4.json
#   bash scripts/infer_with_config.sh configs/val/full_view_stage3_30_60_v4.json --sample_index 3 --out outputs/demo.jpg
#   CONFIG=configs/val/debug_v2_100steps.json bash scripts/infer_with_config.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -gt 0 && ( "$1" == *.json || "$1" == *.yaml || "$1" == *.yml ) ]]; then
  CONFIG_PATH="$1"
  shift
else
  CONFIG_PATH="${CONFIG:-configs/val/full_view_stage3_30_60_v4.json}"
fi

source scripts/00_local_flux_env.sh
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[INFER] CONFIG=${CONFIG_PATH}"
echo "[INFER] Extra CLI overrides: $*"

python src/infer_flux_multiview.py --config "${CONFIG_PATH}" "$@"
