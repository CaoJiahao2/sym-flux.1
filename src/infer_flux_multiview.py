from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from flux.sampling import get_schedule
from flux.util import load_ae
from src.local_text_encoders import load_local_text_encoders

from src.models.flux_multiview_loader import load_multiview_flux
from src.training.flux_train_utils import encode_prompts, make_img_ids, pack_latents, unpack_latents


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="flux-dev")
    p.add_argument("--mv_ckpt", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--camera_json", default=None, help="JSON file containing list [V,12] or object with key 'extrinsics'.")
    p.add_argument("--num_views", type=int, default=4)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mv_adapter_dim", type=int, default=512)
    p.add_argument("--mv_attn_mode", choices=["same_token", "full_view"], default="same_token")
    p.add_argument("--no_mv_timestep_modulation", action="store_true")
    p.add_argument("--inject_single_blocks", action="store_true")
    p.add_argument("--single_block_stride", type=int, default=4)
    p.add_argument("--hf_download", action="store_true")
    p.add_argument("--out", default="outputs/flux_mv_demo.jpg")
    return p.parse_args()


def load_manifest_item(path: str, index: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"sample_index={index} out of range for {path}")


def load_camera_array(args) -> tuple[torch.Tensor, str]:
    prompt = args.prompt
    if args.manifest:
        item = load_manifest_item(args.manifest, args.sample_index)
        cams = item["extrinsics"][: args.num_views]
        if prompt is None:
            prompt = item.get("prompt", "a realistic 3D-rendered scene with a character performing an action")
        return torch.tensor(cams, dtype=torch.float32), prompt

    if args.camera_json:
        with open(args.camera_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        cams = obj.get("extrinsics", obj) if isinstance(obj, dict) else obj
        return torch.tensor(cams[: args.num_views], dtype=torch.float32), prompt

    raise ValueError("Provide --manifest or --camera_json")


# def make_noise(num_views: int, height: int, width: int, device, dtype, seed: int) -> torch.Tensor:
#     latent_h = 2 * math.ceil(height / 16)
#     latent_w = 2 * math.ceil(width / 16)
#     g = torch.Generator(device="cpu").manual_seed(seed)
#     base = torch.randn((1, 16, latent_h, latent_w), generator=g, dtype=torch.float32)
#     indep = torch.randn((num_views, 16, latent_h, latent_w), generator=g, dtype=torch.float32)
#     x = 0.97 * base.repeat(num_views, 1, 1, 1) + 0.03 * indep
#     return x.to(device=device, dtype=dtype)
def make_noise(num_views: int, height: int, width: int, device, dtype, seed: int) -> torch.Tensor:
    """Create independent initial Gaussian noise for each view."""
    latent_h = 2 * math.ceil(height / 16)
    latent_w = 2 * math.ceil(width / 16)
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((num_views, 16, latent_h, latent_w), generator=g, dtype=torch.float32)
    return x.to(device=device, dtype=dtype)


def tensor_to_pil_grid(x: torch.Tensor, labels: list[str] | None = None) -> Image.Image:
    x = (x.detach().float().cpu().clamp(-1, 1) + 1) / 2
    imgs = []
    for i in range(x.shape[0]):
        arr = (x[i].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        img = Image.fromarray(arr)
        if labels:
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 110, 26], fill=(255, 255, 255))
            draw.text((6, 6), labels[i], fill=(0, 0, 0))
        imgs.append(img)
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * len(imgs), h), color=(255, 255, 255))
    for i, img in enumerate(imgs):
        grid.paste(img, (i * w, 0))
    return grid


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    cameras, prompt = load_camera_array(args)
    if prompt is None:
        raise ValueError("Provide --prompt, or use a manifest item containing 'prompt'.")
    cameras = cameras[: args.num_views][None].to(device=device, dtype=dtype)  # [1,V,12]

    t5, clip = load_local_text_encoders(device=device, dtype=dtype, max_length=512)
    try:
        ae = load_ae(args.model_name, device=device, hf_download=args.hf_download).eval()
    except TypeError:
        ae = load_ae(args.model_name, device=device).eval()
    model = load_multiview_flux(
        name=args.model_name,
        device=device,
        dtype=dtype,
        hf_download=args.hf_download,
        mv_adapter_dim=args.mv_adapter_dim,
        inject_single_blocks=args.inject_single_blocks,
        single_block_stride=args.single_block_stride,
        mv_attn_mode=args.mv_attn_mode,
        mv_use_timestep_modulation=not args.no_mv_timestep_modulation,
        mv_ckpt=args.mv_ckpt,
    ).eval()

    x = make_noise(args.num_views, args.height, args.width, device, dtype, args.seed)
    latent_h, latent_w = x.shape[-2], x.shape[-1]
    img = pack_latents(x)
    img_ids = make_img_ids(args.num_views, latent_h, latent_w, device, dtype)
    txt, txt_ids, y = encode_prompts(t5, clip, [prompt] * args.num_views, device, dtype)
    guidance = torch.full((args.num_views,), args.guidance, device=device, dtype=dtype)

    timesteps = get_schedule(args.num_steps, img.shape[1], shift=(args.model_name != "flux-schnell"))
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((args.num_views,), float(t_curr), device=device, dtype=dtype)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t_vec,
            y=y,
            guidance=guidance,
            cameras=cameras,
            num_views=args.num_views,
        )
        img = img + (float(t_prev) - float(t_curr)) * pred

    latents = unpack_latents(img, latent_h, latent_w)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = ae.decode(latents.float())
    labels = [f"view {i+1}" for i in range(args.num_views)]
    grid = tensor_to_pil_grid(out, labels=labels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
