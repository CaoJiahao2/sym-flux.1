import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


# 强制 transformers / huggingface 离线
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


CAPTION_PROMPT = """
You are generating a scene-level caption for training a text-to-image model.

The input image is a preview grid sampled from one 3D scene, containing multiple camera views and multiple time frames.

Write exactly one concise English sentence describing the shared scene content.

Rules:
- Maximum 40 words.
- Mention the main subject, action, environment, and important visual attributes.
- Do not mention camera, viewpoint, view, grid, frame, collage, multi-view, left, right, front, back, close-up, wide shot, angle, cam01, cam05, or cam10.
- Do not say "the image shows", "the scene shows", or "this scene shows".
- Do not describe different views separately.
- Output only the final caption.
""".strip()


REWRITE_PROMPT_TEMPLATE = """
Rewrite the following caption as one scene-level English caption for text-to-image training.

Rules:
- Maximum 40 words.
- Keep only the shared scene content: subject, action, environment, and important visual attributes.
- Remove camera/view/grid/frame/collage/left/right/front/back/close-up/wide-shot/angle information.
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

    # 去掉编号、引号、markdown
    text = re.sub(r"^[-*\d\.\)\s]+", "", text)
    text = text.strip(" \"'`“”‘’")

    # 去掉常见不合适开头
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

    # 只保留第一句，避免模型输出解释
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


def truncate_to_40_words(text: str, max_words: int = 40) -> str:
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
    """
    兼容不同版本 transformers / Qwen chat template。
    尽量关闭 thinking；如果当前 processor 不支持该参数，则自动 fallback。
    """
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

    # 优先使用 qwen_vl_utils；如果不可用，则 fallback 到 PIL image
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
    inputs = prepare_multimodal_inputs(processor, image_path, CAPTION_PROMPT, device)
    caption = generate_text(model, processor, inputs, max_new_tokens=96)

    # 自动 rewrite，不人工筛选
    need_rewrite = (
        word_count(caption) > max_words
        or contains_banned_terms(caption)
        or word_count(caption) < 5
    )

    if need_rewrite:
        rewrite_prompt = REWRITE_PROMPT_TEMPLATE.format(caption=caption)
        rewrite_inputs = prepare_text_inputs(processor, rewrite_prompt, device)
        caption = generate_text(model, processor, rewrite_inputs, max_new_tokens=80)

    # 最终强制不超过 40 words
    if word_count(caption) > max_words:
        caption = truncate_to_40_words(caption, max_words=max_words)

    # 如果模型仍然输出空句，给一个保底 caption
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--preview_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--max_words", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    preview_dir = Path(args.preview_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")

    if not preview_dir.exists():
        raise FileNotFoundError(f"Preview dir not found: {preview_dir}")

    image_paths = sorted(
        preview_dir.glob("scene*.jpg"),
        key=lambda p: int(p.stem.replace("scene", "")) if p.stem.replace("scene", "").isdigit() else p.stem,
    )

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    captions = load_existing(out_path) if args.resume else {}

    print(f"Loading local model from: {model_path}")
    print("Offline mode: local_files_only=True")

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    # 取模型所在 device；device_map=auto 时 inputs 放到第一个参数 device 通常可行
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total = len(image_paths)
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
            # 不人工筛选 bad case；失败时自动写保底 caption，保证流程不中断。
            print(f"[WARN] caption failed for {scene_id}: {repr(e)}")
            caption = "a realistic animated subject moving through a 3D rendered environment."

        captions[scene_id] = caption
        done += 1

        # 每 20 条保存一次，防止中断丢失
        if done % 20 == 0:
            atomic_save(captions, out_path)

    atomic_save(captions, out_path)

    # 自动统计，不进入人工筛选流程
    lengths = [word_count(v) for v in captions.values()]
    too_long = sum(x > args.max_words for x in lengths)
    banned = sum(contains_banned_terms(v) for v in captions.values())

    print(f"Saved: {out_path}")
    print(f"Total captions: {len(captions)}")
    print(f"Max words: {max(lengths) if lengths else 0}")
    print(f"Too long captions: {too_long}")
    print(f"Captions containing banned terms: {banned}")


if __name__ == "__main__":
    main()