#!/usr/bin/env bash
set -euo pipefail

# 查看训练 loss 曲线：
#   bash scripts/full_view/40_tensorboard.sh
# 然后浏览器打开 http://127.0.0.1:6006

LOGDIR="${LOGDIR:-outputs}"
PORT="${PORT:-6006}"
tensorboard --logdir "${LOGDIR}" --port "${PORT}" --bind_all
