# scripts/02_caption_scene_level_qwen35.py

import argparse
import json
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--preview_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_words", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    #TODO
    parser.add_argument(
    "--start_scene",
    type=int,
    default=1,
    help="Only process scenes whose numeric id is >= start_scene, e.g. 2781 for scene2781.",
    )
    #
    parser.add_argument("--gpu_id", type=str, default="0")

    return parser.parse_args()


args = parse_args()

# 必须在 import torch / transformers 之前设置
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# 强制离线，禁止联网下载模型权重
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


CAPTION_PROMPT = """
You are generating a high-quality scene-level caption for training a text-to-image model.

The input image is a preview grid sampled from one 3D scene, containing multiple camera views and multiple time frames.

Write exactly one detailed English sentence describing the shared scene content.

Rules:
- Maximum 50 words.
- Describe the main subject, action, environment, appearance, clothing or texture if visible, lighting, mood, and notable objects when they are clearly shared across the scene.
- Focus on content that remains consistent across the whole scene.
- Use natural, vivid, training-friendly language.
- Do not mention camera, viewpoint, view, grid, frame, collage, multi-view, left, right, front, back, close-up, wide shot, angle, cam01, cam05, or cam10.
- Do not say "the image shows", "the scene shows", or "this scene shows".
- Do not describe different views separately.
- Output only the final caption.
""".strip()


REWRITE_PROMPT_TEMPLATE = """
Rewrite the following caption as one high-quality scene-level English caption for text-to-image training.

Rules:
- Maximum 50 words.
- Keep only the shared scene content.
- Preserve or improve details about the subject, action, environment, appearance, clothing or texture if visible, lighting, mood, and notable objects.
- Remove camera/view/grid/frame/collage/left/right/front/back/close-up/wide-shot/angle information.
- Make the sentence natural, descriptive, and training-friendly.
- Output only the rewritten caption.

Caption:
{caption}
""".strip()


BANNED_PATTERNS = [
    r"\bcamera\b",
    r"\bviewpoint\b",
    r"\bviews?\b",
    r"\bgrid\b",
    r"\bframes?\b",
    r"\bcollage\b",
    r"\bmulti[- ]?view\b",
    r"\bleft\b",
    r"\bright\b",
    r"\bfront\b",
    r"\bback\b",
    r"\bclose[- ]?up\b",
    r"\bwide shot\b",
    r"\bangle\b",
    r"\bcam\d+\b",
    r"\bthe image shows\b",
    r"\bthis image shows\b",
    r"\bthe scene shows\b",
    r"\bthis scene shows\b",
    r"\bthe image depicts\b",
    r"\bthis image depicts\b",
]


def remove_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def normalize_caption(text: str) -> str:
    text = remove_thinking(text)
    text = text.strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"^[-*\d\.\)\s]+", "", text)
    text = text.strip(" \"'`“”‘’")

    prefixes = [
        "The image shows ",
        "This image shows ",
        "The scene shows ",
        "This scene shows ",
        "The image depicts ",
        "This image depicts ",
        "It shows ",
        "It depicts ",
    ]

    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    parts = re.split(r"(?<=[.!?])\s+", text)
    if parts:
        text = parts[0].strip()

    if len(text) > 1:
        text = text[0].lower() + text[1:]

    if text and text[-1] not in ".!?":
        text += "."

    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def truncate_to_50_words(text: str, max_words: int = 50) -> str:
    words = re.findall(r"\b[\w'-]+\b|[^\w\s]", text)
    count = 0
    kept = []

    for tok in words:
        if re.match(r"\b[\w'-]+\b", tok):
            count += 1
        if count > max_words:
            break
        kept.append(tok)

    out = " ".join(kept)
    out = re.sub(r"\s+([,.!?;:])", r"\1", out)
    out = out.strip(" ,;:")

    if out and out[-1] not in ".!?":
        out += "."

    return out


def contains_banned_terms(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in BANNED_PATTERNS)


def apply_chat_template(processor, messages):
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        pass

    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        pass

    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_multimodal_inputs(processor, image_path: Path, prompt: str, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = apply_chat_template(processor, messages)

    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

    return inputs.to(device)


def prepare_text_inputs(processor, prompt: str, device):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = apply_chat_template(processor, messages)

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )

    return inputs.to(device)


@torch.no_grad()
def generate_text(model, processor, inputs, max_new_tokens: int = 96) -> str:
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    input_ids = inputs["input_ids"]
    generated_ids = generated_ids[:, input_ids.shape[1]:]

    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return normalize_caption(text)


@torch.no_grad()
def caption_image(model, processor, image_path: Path, device, max_words: int) -> str:
    inputs = prepare_multimodal_inputs(
        processor=processor,
        image_path=image_path,
        prompt=CAPTION_PROMPT,
        device=device,
    )

    caption = generate_text(
        model=model,
        processor=processor,
        inputs=inputs,
        max_new_tokens=128,
    )

    need_rewrite = (
        word_count(caption) > max_words
        or contains_banned_terms(caption)
        or word_count(caption) < 8
    )

    if need_rewrite:
        rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(caption=caption)
        rewrite_inputs = prepare_text_inputs(
            processor=processor,
            prompt=rewrite_prompt,
            device=device,
        )

        caption = generate_text(
            model=model,
            processor=processor,
            inputs=rewrite_inputs,
            max_new_tokens=96,
        )

    if word_count(caption) > max_words:
        caption = truncate_to_50_words(caption, max_words=max_words)

    if word_count(caption) < 3:
        caption = "a realistic animated subject moving through a 3D rendered environment."

    return caption


def load_existing(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main():
    model_path = Path(args.model_path)
    preview_dir = Path(args.preview_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")

    if not preview_dir.exists():
        raise FileNotFoundError(f"Preview dir not found: {preview_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check NVIDIA driver, CUDA, and PyTorch installation."
        )

    device = torch.device("cuda:0")

    print("=" * 80)
    print(f"Physical GPU id requested: {args.gpu_id}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Using logical device: {device}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Loading local model from: {model_path}")
    print("Offline mode: local_files_only=True")
    print("=" * 80)

    # TODO
    def scene_number(path: Path):
        name = path.stem.replace("scene", "")
        return int(name) if name.isdigit() else -1


    image_paths = sorted(
        preview_dir.glob("scene*.jpg"),
        key=scene_number,
    )

    # 临时断点续跑：只处理 scene2781 及之后
    image_paths = [
        p for p in image_paths
        if scene_number(p) >= args.start_scene
    ]

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    print(f"Start scene: scene{args.start_scene}")
    print(f"Number of preview images to process after filtering: {len(image_paths)}")
    if len(image_paths) > 0:
        print(f"First scene to process: {image_paths[0].stem}")
        print(f"Last scene to process: {image_paths[-1].stem}")
    # TODO

    captions = load_existing(out_path) if args.resume else {}

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()

    done = 0

    for image_path in tqdm(image_paths, desc="Captioning scenes"):
        scene_id = image_path.stem

        if args.resume and scene_id in captions:
            continue

        try:
            caption = caption_image(
                model=model,
                processor=processor,
                image_path=image_path,
                device=device,
                max_words=args.max_words,
            )
        except Exception as e:
            print(f"[WARN] caption failed for {scene_id}: {repr(e)}")
            caption = "a realistic animated subject moving through a 3D rendered environment."

        captions[scene_id] = caption
        done += 1

        if done % 20 == 0:
            atomic_save(captions, out_path)

    atomic_save(captions, out_path)

    lengths = [word_count(v) for v in captions.values()]
    too_long = sum(x > args.max_words for x in lengths)
    banned = sum(contains_banned_terms(v) for v in captions.values())

    print("=" * 80)
    print(f"Saved: {out_path}")
    print(f"Total captions: {len(captions)}")
    print(f"Max words: {max(lengths) if lengths else 0}")
    print(f"Too long captions: {too_long}")
    print(f"Captions containing banned terms: {banned}")
    print("=" * 80)


if __name__ == "__main__":
    main()