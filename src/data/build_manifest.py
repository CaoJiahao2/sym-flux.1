"""Build train/val jsonl manifests for SynCamVideo-Dataset.

The script reads SynCamVideo-Dataset and writes jsonl files outside the dataset
folder. It never changes files under SynCamVideo-Dataset.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

# Allow running both as `python src/data/build_manifest.py` and as module.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from camera_utils import (  # noqa: E402
    load_scene_extrinsics,
    max_pairwise_rotation_angle_deg,
    normalize_extrinsics_to_anchor,
)


DEFAULT_PROMPT = "a realistic 3D-rendered scene with a character performing an action"


def load_captions(path: str | None) -> Dict[str, str]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] captions file not found: {p}; using fallback prompt for all scenes.")
        return {}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"captions.json must be a dict: scene_id -> prompt, got {type(obj)}")
    return {str(k): str(v) for k, v in obj.items()}


def list_scene_dirs(dataset_root: Path, split: str, aperture: str) -> List[Path]:
    split_root = dataset_root / split / aperture
    if not split_root.exists():
        raise FileNotFoundError(f"Split/aperture directory does not exist: {split_root}")
    return sorted([p for p in split_root.iterdir() if p.is_dir() and p.name.startswith("scene")])


def frame_sort_key(frame_key: str) -> int:
    return int(frame_key.replace("frame", ""))


def select_views(
    all_cams: List[str],
    all_mats: Dict[str, object],
    num_views: int,
    max_angle: float | None,
    rng: random.Random,
    sampling: str,
) -> List[str] | None:
    """Select a camera group from the available cameras.

    If num_views <= 0 or num_views >= number of cameras, return all cameras.
    """
    if num_views <= 0 or num_views >= len(all_cams):
        return list(all_cams)

    combos = list(itertools.combinations(all_cams, num_views))
    if sampling == "random":
        rng.shuffle(combos)

    valid = []
    for combo in combos:
        mats = [all_mats[c] for c in combo]
        angle = max_pairwise_rotation_angle_deg(mats)
        if max_angle is None or angle <= max_angle:
            valid.append((angle, list(combo)))

    if not valid:
        return None

    if sampling == "first":
        valid.sort(key=lambda x: (x[0], x[1]))
        return valid[0][1]

    return valid[0][1]


def build_manifest(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    captions = load_captions(args.captions)
    scenes = list_scene_dirs(dataset_root, args.split, args.aperture)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    rng = random.Random(args.seed)
    total = 0
    skipped = 0

    cam_filter = None
    if args.cams.strip():
        cam_filter = [c.strip() for c in args.cams.split(",") if c.strip()]

    with out_path.open("w", encoding="utf-8") as fout:
        for scene_dir in scenes:
            scene_id = scene_dir.name
            camera_json = scene_dir / "cameras" / "camera_extrinsics.json"
            video_dir = scene_dir / "videos"

            if not camera_json.exists():
                print(f"[WARN] missing camera json: {camera_json}")
                skipped += 1
                continue

            extrinsics_by_frame = load_scene_extrinsics(camera_json)
            frame_keys = sorted(extrinsics_by_frame.keys(), key=frame_sort_key)

            if args.frame_stride > 1:
                frame_keys = [fk for fk in frame_keys if frame_sort_key(fk) % args.frame_stride == 0]

            prompt = captions.get(scene_id, DEFAULT_PROMPT)

            for frame_key in frame_keys:
                frame_idx = frame_sort_key(frame_key)
                cam_dict = extrinsics_by_frame[frame_key]
                all_cams = sorted(cam_dict.keys())

                if cam_filter is not None:
                    all_cams = [c for c in cam_filter if c in cam_dict]

                # Keep only cameras that have corresponding mp4 files.
                all_cams = [c for c in all_cams if (video_dir / f"{c}.mp4").exists()]

                if len(all_cams) == 0:
                    skipped += 1
                    continue

                selected_cams = select_views(
                    all_cams=all_cams,
                    all_mats=cam_dict,
                    num_views=args.num_views,
                    max_angle=args.max_angle,
                    rng=rng,
                    sampling=args.sampling,
                )
                if selected_cams is None:
                    skipped += 1
                    continue

                selected_mats = [cam_dict[c] for c in selected_cams]
                rel_extrinsics = normalize_extrinsics_to_anchor(
                    selected_mats,
                    anchor_index=0,
                    convention=args.convention,
                )
                angle_max = max_pairwise_rotation_angle_deg(selected_mats)

                item = {
                    "dataset_root": str(dataset_root),
                    "split": args.split,
                    "aperture": args.aperture,
                    "scene": scene_id,
                    "frame_idx": frame_idx,
                    "frame_key": frame_key,
                    "cams": selected_cams,
                    "videos": [str(video_dir / f"{c}.mp4") for c in selected_cams],
                    "extrinsics": rel_extrinsics.astype(float).tolist(),
                    "extrinsics_convention": args.convention,
                    "anchor_cam": selected_cams[0],
                    "max_pairwise_rotation_deg": angle_max,
                    "prompt": prompt,
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                total += 1

    print(f"[OK] wrote manifest: {out_path}")
    print(f"[OK] samples: {total}; skipped: {skipped}; scenes considered: {len(scenes)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "val"])
    parser.add_argument("--aperture", type=str, default="f24_aperture5")
    parser.add_argument("--captions", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_scenes", type=int, default=0, help="0 means all scenes")
    parser.add_argument("--frame_stride", type=int, default=8)
    parser.add_argument("--num_views", type=int, default=0, help="0 means keep all available cameras")
    parser.add_argument("--cams", type=str, default="", help="Optional comma list, e.g. cam01,cam02,...")
    parser.add_argument("--max_angle", type=float, default=None, help="Optional max pairwise rotation angle in degrees")
    parser.add_argument("--sampling", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--convention", type=str, default="w2c", choices=["w2c", "c2w", "none"])
    return parser.parse_args()


if __name__ == "__main__":
    build_manifest(parse_args())
