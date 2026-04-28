import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm


def read_frame(video_path: Path, frame_idx: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count: {video_path}")

    frame_idx = min(frame_idx, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def make_grid(images, labels, thumb_size: int = 256) -> Image.Image:
    n_rows = 3
    n_cols = 5
    label_h = 28

    grid = Image.new(
        "RGB",
        (n_cols * thumb_size, n_rows * (thumb_size + label_h)),
        "white",
    )

    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // n_cols
        col = i % n_cols

        img = img.resize((thumb_size, thumb_size), Image.BICUBIC)

        cell = Image.new("RGB", (thumb_size, thumb_size + label_h), "white")
        draw = ImageDraw.Draw(cell)
        draw.text((6, 6), label, fill=(0, 0, 0))
        cell.paste(img, (0, label_h))

        grid.paste(cell, (col * thumb_size, row * (thumb_size + label_h)))

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_scenes", type=int, default=-1)
    parser.add_argument("--thumb_size", type=int, default=256)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    scene_root = dataset_root / args.split / "f24_aperture5"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scene_root.exists():
        raise FileNotFoundError(f"Scene root not found: {scene_root}")

    scenes = sorted(
        [p for p in scene_root.iterdir() if p.is_dir() and p.name.startswith("scene")],
        key=lambda p: int(p.name.replace("scene", "")) if p.name.replace("scene", "").isdigit() else p.name,
    )

    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    cams = ["cam01", "cam05", "cam10"]
    frames = [0, 20, 40, 60, 80]

    ok_count = 0
    skip_count = 0

    for scene_dir in tqdm(scenes, desc=f"Making previews: {args.split}"):
        scene_id = scene_dir.name
        out_path = out_dir / f"{scene_id}.jpg"

        if out_path.exists():
            ok_count += 1
            continue

        images = []
        labels = []
        failed = False

        for cam in cams:
            video_path = scene_dir / "videos" / f"{cam}.mp4"
            if not video_path.exists():
                print(f"[WARN] missing video: {video_path}")
                failed = True
                break

            for frame_idx in frames:
                try:
                    img = read_frame(video_path, frame_idx)
                    images.append(img)
                    labels.append(f"{cam} / f{frame_idx}")
                except Exception as e:
                    print(f"[WARN] {scene_id} {cam} frame {frame_idx}: {e}")
                    failed = True
                    break

            if failed:
                break

        if failed or len(images) != len(cams) * len(frames):
            skip_count += 1
            continue

        grid = make_grid(images, labels, thumb_size=args.thumb_size)
        grid.save(out_path, quality=95)
        ok_count += 1

    print(f"Done. ok={ok_count}, skipped={skip_count}, out_dir={out_dir}")


if __name__ == "__main__":
    main()