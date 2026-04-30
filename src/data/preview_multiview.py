"""Create a horizontal contact sheet for cam01-cam10 at one scene/frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import decord
from PIL import Image, ImageDraw, ImageFont


def read_frame(video_path: Path, frame_idx: int, resize: int | None = None) -> Image.Image:
    vr = decord.VideoReader(str(video_path))
    if frame_idx >= len(vr):
        raise IndexError(f"frame_idx={frame_idx} out of range for {video_path}, len={len(vr)}")
    frame = vr[frame_idx].asnumpy()
    img = Image.fromarray(frame).convert("RGB")
    if resize is not None:
        img = img.resize((resize, resize), Image.Resampling.BICUBIC)
    return img


def make_contact_sheet(images: list[Image.Image], labels: list[str], title: str, out: Path) -> None:
    assert len(images) == len(labels)
    label_h = 30
    title_h = 42
    w, h = images[0].size
    sheet = Image.new("RGB", (w * len(images), h + label_h + title_h), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    draw.rectangle((0, 0, sheet.width, title_h), fill=(245, 245, 245))
    draw.text((10, 9), title, fill=(0, 0, 0), font=title_font)

    y0 = title_h
    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * w
        sheet.paste(img, (x, y0 + label_h))
        draw.rectangle((x, y0, x + w, y0 + label_h), fill=(230, 230, 230))
        draw.text((x + 8, y0 + 5), label, fill=(0, 0, 0), font=font)

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"[OK] saved preview: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--aperture", type=str, default="f24_aperture5")
    parser.add_argument("--scene", type=str, default="scene1")
    parser.add_argument("--frame_idx", type=int, default=40)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--out", type=str, default="outputs/preview_scene1_frame40_cam01_cam10.jpg")
    parser.add_argument("--cams", type=str, default="cam01,cam02,cam03,cam04,cam05,cam06,cam07,cam08,cam09,cam10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    scene_dir = dataset_root / args.split / args.aperture / args.scene
    video_dir = scene_dir / "videos"
    camera_json = scene_dir / "cameras" / "camera_extrinsics.json"

    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    if not camera_json.exists():
        raise FileNotFoundError(f"Camera extrinsics not found: {camera_json}")

    with camera_json.open("r", encoding="utf-8") as f:
        cam_meta = json.load(f)
    frame_key = f"frame{args.frame_idx}"
    if frame_key not in cam_meta:
        raise KeyError(f"{frame_key} not found in {camera_json}")

    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    images = []
    labels = []
    for cam in cams:
        video_path = video_dir / f"{cam}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        images.append(read_frame(video_path, args.frame_idx, resize=args.resize))
        labels.append(cam)

    title = f"{args.split}/{args.aperture}/{args.scene} | frame{args.frame_idx} | cam01-cam10"
    make_contact_sheet(images, labels, title, Path(args.out))


if __name__ == "__main__":
    main()
