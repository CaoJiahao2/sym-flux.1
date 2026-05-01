#!/usr/bin/env bash
set -euo pipefail

# 顺序训练三个角度阶段：[0,30] -> [15,45] -> [30,60]。
# 后一阶段会默认从前一阶段的 mv_adapter_last.pt 继续。
#
# 常用：
#   GPU_IDS=0 MV_ARCH=adapter bash scripts/full_view/23_train_all_full_view_stages.sh
#   GPU_IDS=0 MV_ARCH=full_hidden GRAD_ACCUM=16 bash scripts/full_view/23_train_all_full_view_stages.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

bash scripts/full_view/20_train_full_view_0_30.sh
bash scripts/full_view/21_train_full_view_15_45.sh
bash scripts/full_view/22_train_full_view_30_60.sh
