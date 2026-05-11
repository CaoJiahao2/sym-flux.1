#!/usr/bin/env bash
set -euo pipefail

# Config-first inference entry.
# Usage:
#   bash scripts/infer.sh
#   bash scripts/infer.sh configs/val/full_view_stage3_30_60_v4.json
#   bash scripts/infer.sh configs/val/default.json --sample_index 3 --out outputs/demo.jpg
#   CONFIG=configs/val/default.json bash scripts/infer.sh --seed 123
#
# All inference defaults come from the config. CLI flags after the config override config values.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${CONFIG:-configs/val/default.json}"
if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_PATH="$1"
  shift
fi

if [[ -f scripts/00_local_flux_env.sh ]]; then
  source scripts/00_local_flux_env.sh
fi
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[INFER] CONFIG=${CONFIG_PATH}"
if [[ $# -gt 0 ]]; then
  echo "[INFER] CLI overrides: $*"
fi

python src/infer_flux_multiview.py --config "${CONFIG_PATH}" "$@"
