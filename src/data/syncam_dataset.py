from __future__ import annotations

import json
import random
from pathlib import Path

import decord
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


decord.bridge.set_bridge("torch")


class SynCamImageGroupDataset(Dataset):
    """Read manifest jsonl and return synchronized multi-view image groups.

    Each item contains:
      pixel_values: [V,3,H,W] in [-1,1]
      cameras:      [V,12]
      prompt:       str
    """

    def __init__(
        self,
        manifest_path: str | Path,
        resolution: int = 512,
        num_views: int | None = None,
        random_view_subset: bool = False,
    ) -> None:
        self.items = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))
        self.resolution = resolution
        self.num_views = num_views
        self.random_view_subset = random_view_subset
        self.tf = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def _select_view_indices(self, item: dict) -> list[int]:
        n = len(item["videos"])
        if self.num_views is None or self.num_views <= 0 or self.num_views >= n:
            return list(range(n))
        if self.random_view_subset:
            ids = list(range(n))
            # Keep anchor view 0 and sample the rest. This matches relative extrinsics anchored to view 0.
            rest = ids[1:]
            random.shuffle(rest)
            return [0] + sorted(rest[: self.num_views - 1])
        return list(range(self.num_views))

    def _read_frame(self, video_path: str, frame_idx: int) -> torch.Tensor:
        vr = decord.VideoReader(video_path)
        frame = vr[frame_idx].numpy()
        img = Image.fromarray(frame).convert("RGB")
        return self.tf(img)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        frame_idx = int(item["frame_idx"])
        view_ids = self._select_view_indices(item)

        images = [self._read_frame(item["videos"][i], frame_idx) for i in view_ids]
        cameras = [item["extrinsics"][i] for i in view_ids]
        cams = [item.get("cams", [])[i] for i in view_ids] if "cams" in item else [str(i) for i in view_ids]

        return {
            "pixel_values": torch.stack(images, dim=0),
            "cameras": torch.tensor(cameras, dtype=torch.float32),
            "prompt": item.get("prompt", "a realistic 3D-rendered scene with a character performing an action"),
            "scene": item.get("scene", ""),
            "frame_idx": frame_idx,
            "cams": cams,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "cameras": torch.stack([b["cameras"] for b in batch], dim=0),
        "prompts": [b["prompt"] for b in batch],
        "meta": [
            {"scene": b.get("scene", ""), "frame_idx": b.get("frame_idx", -1), "cams": b.get("cams", [])}
            for b in batch
        ],
    }
