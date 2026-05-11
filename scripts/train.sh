#!/usr/bin/env bash
set -euo pipefail

# Config-first training entry.
# Usage:
#   bash scripts/train.sh
#   bash scripts/train.sh configs/train/full_view_stage1_0_30_v4.json
#   bash scripts/train.sh configs/train/default.json --max_steps 20
#   CONFIG=configs/train/default.json bash scripts/train.sh --max_steps 20
#
# Policy:
#   - All training hyperparameters are read from the config by default.
#   - CLI flags after the config override config values.
#   - If config contains a non-empty output_dir, use it.
#   - If config omits output_dir and no CLI --output_dir is provided, create a stable auto output_dir.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${CONFIG:-configs/train/default.json}"
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

json_value() {
  python - "$CONFIG_PATH" "$1" "${2:-}" <<'PY'
import json, sys
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
value = cfg.get(key, default)
if value is None:
    value = default
print(value)
PY
}

has_cli_output_dir=0
for arg in "$@"; do
  if [[ "$arg" == "--output_dir" || "$arg" == --output_dir=* ]]; then
    has_cli_output_dir=1
    break
  fi
done

CONFIG_OUTPUT_DIR="$(json_value output_dir "")"
OUTPUT_ARGS=()
EFFECTIVE_OUTPUT_DIR="${CONFIG_OUTPUT_DIR}"

if [[ -z "${CONFIG_OUTPUT_DIR}" && "${has_cli_output_dir}" -eq 0 ]]; then
  MV_ATTN_MODE="$(json_value mv_attn_mode full_view)"
  MV_ARCH="$(json_value mv_arch adapter)"
  MIN_ANGLE="$(json_value min_angle 0)"
  MAX_ANGLE="$(json_value max_angle 30)"
  MAX_STEPS="$(json_value max_steps 5000)"
  NUM_VIEWS="$(json_value num_views 4)"
  MV_ADAPTER_DIM="$(json_value mv_adapter_dim 512)"
  GRAD_ACCUM="$(json_value grad_accum 1)"
  RUN_NAME="${MV_ATTN_MODE}_angle_${MIN_ANGLE}-${MAX_ANGLE}_${MV_ARCH}_steps${MAX_STEPS}_views${NUM_VIEWS}_dim${MV_ADAPTER_DIM}_grad${GRAD_ACCUM}"
  EFFECTIVE_OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
  OUTPUT_ARGS=(--output_dir "${EFFECTIVE_OUTPUT_DIR}")
fi

if [[ -n "${EFFECTIVE_OUTPUT_DIR}" ]]; then
  mkdir -p "${EFFECTIVE_OUTPUT_DIR}"
fi

echo "[TRAIN] CONFIG=${CONFIG_PATH}"
if [[ -n "${EFFECTIVE_OUTPUT_DIR}" ]]; then
  echo "[TRAIN] OUTPUT_DIR=${EFFECTIVE_OUTPUT_DIR}"
else
  echo "[TRAIN] OUTPUT_DIR is provided by CLI override"
fi
if [[ $# -gt 0 ]]; then
  echo "[TRAIN] CLI overrides: $*"
fi

python src/train_flux_multiview.py --config "${CONFIG_PATH}" "${OUTPUT_ARGS[@]}" "$@"
