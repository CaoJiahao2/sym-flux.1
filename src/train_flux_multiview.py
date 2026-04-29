from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# motified
# from flux.util import load_ae, load_clip, load_t5
from flux.util import load_ae
from src.local_text_encoders import load_local_text_encoders

from src.data.syncam_dataset import SynCamImageGroupDataset, collate_fn
from src.models.flux_multiview import extract_mv_state_dict
from src.models.flux_multiview_loader import load_multiview_flux
from src.training.flux_train_utils import (
    count_trainable_params,
    encode_prompts,
    expand_prompts_for_views,
    make_img_ids,
    pack_latents,
    sample_flow_timesteps,
    shared_view_noise_like,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="flux-dev")
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_views", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--mixed_precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--mv_adapter_dim", type=int, default=512)
    p.add_argument("--mv_dropout", type=float, default=0.0)
    p.add_argument("--inject_single_blocks", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--noise_share_ratio", type=float, default=0.75)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--hf_download", action="store_true")
    p.add_argument("--resume_mv_ckpt", default=None, help="Optional adapter checkpoint to continue training from.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.mixed_precision]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(Path(args.output_dir) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("Loading local text encoders and AE")
    t5, clip = load_local_text_encoders(device=device, dtype=dtype, max_length=512)

    try:
        ae = load_ae(args.model_name, device=device, hf_download=args.hf_download).eval().requires_grad_(False)
    except TypeError:
        ae = load_ae(args.model_name, device=device).eval().requires_grad_(False)
    
    print("Loading FLUX MultiView transformer")
    model = load_multiview_flux(
        name=args.model_name,
        device=device,
        dtype=dtype,
        hf_download=args.hf_download,
        mv_adapter_dim=args.mv_adapter_dim,
        mv_dropout=args.mv_dropout,
        inject_single_blocks=args.inject_single_blocks,
        mv_ckpt=args.resume_mv_ckpt,
    )
    model.freeze_base_model()
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()

    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable/1e6:.2f}M / total {total/1e9:.2f}B")

    dataset = SynCamImageGroupDataset(
        args.train_manifest,
        resolution=args.resolution,
        num_views=args.num_views,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.max_steps)

    while global_step < args.max_steps:
        for batch in loader:
            if global_step >= args.max_steps:
                break

            # pixel_values = batch["pixel_values"].to(device=device, dtype=dtype, non_blocking=True)
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32, non_blocking=True)
            cameras = batch["cameras"].to(device=device, dtype=dtype, non_blocking=True)
            prompts = batch["prompts"]

            b, v, c, h, w = pixel_values.shape
            assert v == args.num_views, f"Dataset returned V={v}, expected {args.num_views}"
            pixel_values = pixel_values.reshape(b * v, c, h, w)
            prompts_expanded = expand_prompts_for_views(prompts, v)

            with torch.no_grad():
                # latents = ae.encode(pixel_values)
                latents = ae.encode(pixel_values).to(dtype=dtype)
                noise = shared_view_noise_like(latents, num_views=v, share_ratio=args.noise_share_ratio)
                t = sample_flow_timesteps(b * v, device=device, dtype=dtype)
                t_img = t.view(-1, 1, 1, 1)
                noisy_latents = (1.0 - t_img) * latents + t_img * noise

                img = pack_latents(noisy_latents)
                target = pack_latents(noise - latents)
                img_ids = make_img_ids(b * v, noisy_latents.shape[-2], noisy_latents.shape[-1], device, dtype)
                txt, txt_ids, y = encode_prompts(t5, clip, prompts_expanded, device, dtype)
                guidance = torch.full((b * v,), args.guidance, device=device, dtype=dtype)

            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=t,
                y=y,
                guidance=guidance,
                cameras=cameras,
                num_views=v,
            )
            loss = F.mse_loss(pred.float(), target.float()) / args.grad_accum
            loss.backward()

            running_loss += float(loss.detach().cpu()) * args.grad_accum
            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                avg = running_loss / max(1, args.log_every)
                tqdm.write(f"step={global_step} loss={avg:.6f}")
                running_loss = 0.0

            if global_step > 0 and global_step % args.save_every == 0:
                save_path = Path(args.output_dir) / f"mv_adapter_step_{global_step}.pt"
                torch.save(
                    {
                        "state_dict": extract_mv_state_dict(model),
                        "args": vars(args),
                        "global_step": global_step,
                    },
                    save_path,
                )
                tqdm.write(f"Saved {save_path}")

            global_step += 1
            pbar.update(1)

    final_path = Path(args.output_dir) / "mv_adapter_last.pt"
    torch.save(
        {"state_dict": extract_mv_state_dict(model), "args": vars(args), "global_step": global_step},
        final_path,
    )
    print(f"Saved final adapter: {final_path}")


if __name__ == "__main__":
    main()
