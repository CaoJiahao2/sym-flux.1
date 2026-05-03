#!/usr/bin/env bash
set -euo pipefail

# Generic training entry. Usage:
#   bash scripts/train_with_config.sh configs/train/default.json
#   bash scripts/train_with_config.sh configs/train/default.json --max_steps 20
#
# Output directory policy:
#   1) If the config explicitly contains a non-empty "output_dir", use it.
#   2) Otherwise create the historical auto run directory from key hparams.
#   3) A command-line --output_dir passed in extra args still overrides both,
#      because argparse gives CLI arguments precedence over config defaults.

CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Usage: bash scripts/train_with_config.sh configs/train/xxx.json [extra args]" >&2
  exit 1
fi
shift || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Keep the wrapper self-contained for the README quick-start path.
if [[ -f scripts/00_local_flux_env.sh ]]; then
  source scripts/00_local_flux_env.sh
fi
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

read_json () {
  python - "$CONFIG_PATH" "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

config_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
value = cfg.get(key, default)
if value is None:
    value = default
print(value)
PY
}

read_json_optional () {
  python - "$CONFIG_PATH" "$1" <<'PY'
import json
import sys

config_path, key = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
value = cfg.get(key, None)
if value is None:
    print("")
else:
    print(str(value))
PY
}

MV_ATTN_MODE="$(read_json mv_attn_mode full_view)"
MV_ARCH="$(read_json mv_arch adapter)"
MIN_ANGLE="$(read_json min_angle 0)"
MAX_ANGLE="$(read_json max_angle 30)"
MAX_STEPS="$(read_json max_steps 5000)"
NUM_VIEWS="$(read_json num_views 4)"
MV_ADAPTER_DIM="$(read_json mv_adapter_dim 512)"
GRAD_ACCUM="$(read_json grad_accum 1)"
CONFIG_OUTPUT_DIR="$(read_json_optional output_dir)"

RUN_NAME="${MV_ATTN_MODE}_angle_${MIN_ANGLE}-${MAX_ANGLE}_${MV_ARCH}_steps${MAX_STEPS}_views${NUM_VIEWS}_dim${MV_ADAPTER_DIM}_grad${GRAD_ACCUM}"

OUTPUT_ARGS=()
if [[ -n "${CONFIG_OUTPUT_DIR}" ]]; then
  EFFECTIVE_OUTPUT_DIR="${CONFIG_OUTPUT_DIR}"
  echo "Using output_dir from config: ${EFFECTIVE_OUTPUT_DIR}"
else
  EFFECTIVE_OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
  OUTPUT_ARGS=(--output_dir "${EFFECTIVE_OUTPUT_DIR}")
  echo "Config has no output_dir; using auto output_dir: ${EFFECTIVE_OUTPUT_DIR}"
fi

mkdir -p "${EFFECTIVE_OUTPUT_DIR}"

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "EFFECTIVE_OUTPUT_DIR=${EFFECTIVE_OUTPUT_DIR}"

python src/train_flux_multiview.py \
  --config "${CONFIG_PATH}" \
  "${OUTPUT_ARGS[@]}" \
  "$@"
