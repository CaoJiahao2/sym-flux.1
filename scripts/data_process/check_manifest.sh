#!/usr/bin/env bash
set -euo pipefail

# Run from project root: bash scripts/02_check_manifest.sh
# Checks one manifest item and verifies dataloader shapes.

MANIFEST="${MANIFEST:-data/stride_10_angle_15_train_samples.jsonl}"
RESOLUTION="${RESOLUTION:-512}"
NUM_VIEWS="${NUM_VIEWS:-10}"

python - <<PY
from src.data.syncam_dataset import SynCamImageGroupDataset, collate_fn
from torch.utils.data import DataLoader

manifest = "${MANIFEST}"
resolution = int("${RESOLUTION}")
num_views = int("${NUM_VIEWS}")

ds = SynCamImageGroupDataset(manifest, resolution=resolution, num_views=num_views)
print("dataset length:", len(ds))
item = ds[0]
print("item pixel_values:", tuple(item["pixel_values"].shape))
print("item cameras:", tuple(item["cameras"].shape))
print("scene/frame/cams:", item["scene"], item["frame_idx"], item["cams"])
print("prompt:", item["prompt"])

loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
batch = next(iter(loader))
print("batch pixel_values:", tuple(batch["pixel_values"].shape))
print("batch cameras:", tuple(batch["cameras"].shape))
PY
