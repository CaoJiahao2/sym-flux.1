#!/usr/bin/env bash
set -e

# Generic training entry. Usage:
#   CONFIG=configs/full_view_stage1_0_30_v4.json bash scripts/train_with_config.sh
#   bash scripts/train_with_config.sh configs/debug_v2_100steps.json --max_steps 20
#   bash scripts/train_with_config.sh --max_steps 20   # uses $CONFIG or default config

CONFIG_PATH="$1"
shift || true

if [ -z "$CONFIG_PATH" ]; then
  echo "Usage: bash scripts/train_with_config.sh configs/train/xxx.json [extra args]"
  exit 1
fi

read_json () {
  python - "$CONFIG_PATH" "$1" "$2" <<'PY'
import json
import sys

config_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

value = cfg.get(key, default)
if value is None:
    value = default

print(value)
PY
}

MV_ATTN_MODE=$(read_json mv_attn_mode cross_view)
MV_ARCH=$(read_json mv_arch mvs)
MIN_ANGLE=$(read_json min_angle 0)
MAX_ANGLE=$(read_json max_angle 30)
MAX_STEPS=$(read_json max_steps 5000)
NUM_VIEWS=$(read_json num_views 4)
MV_ADAPTER_DIM=$(read_json mv_adapter_dim 3072)
GRAD_ACCUM=$(read_json gradient_accumulation_steps 1)

RUN_NAME="${MV_ATTN_MODE}_angle_${MIN_ANGLE}-${MAX_ANGLE}_${MV_ARCH}_steps${MAX_STEPS}_views${NUM_VIEWS}_dim${MV_ADAPTER_DIM}_grad${GRAD_ACCUM}"

export OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python src/train_flux_multiview.py \
  --config "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR" \
  "$@"