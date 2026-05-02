from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flux.sampling import get_schedule
from flux.util import load_ae
from src.config_utils import load_config_file, validate_config_keys
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
    unpack_latents,
)


IDENTITY_CAMERA_12 = torch.tensor(
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    dtype=torch.float32,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to a JSON/YAML config. CLI arguments override config values.")
    p.add_argument("--model_name", default="flux-dev")
    p.add_argument("--train_manifest", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_views", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=1000, help="Number of optimizer steps, not micro-steps.")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--mixed_precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--guidance", type=float, default=3.5)

    # MVS config.
    p.add_argument("--mv_arch", choices=["adapter", "full_hidden"], default="adapter",
                   help="adapter: low-dimensional MVS; full_hidden: D=hidden_size MVS copied from FLUX img_attn.")
    p.add_argument("--mv_adapter_dim", type=int, default=512,
                   help="Only used when --mv_arch adapter. full_hidden uses FLUX hidden_size directly.")
    p.add_argument("--mv_dropout", type=float, default=0.0)
    p.add_argument("--mv_attn_mode", choices=["same_token", "full_view"], default="full_view")
    p.add_argument("--no_mv_timestep_modulation", action="store_true")
    p.add_argument("--inject_single_blocks", dest="inject_single_blocks", action="store_true", default=True)
    p.add_argument("--no_inject_single_blocks", dest="inject_single_blocks", action="store_false")
    p.add_argument("--single_block_stride", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # Training uses independent Gaussian noise by design. Keep this deprecated
    # flag for backward-compatible scripts, but it is intentionally ignored.
    p.add_argument("--noise_share_ratio", type=float, default=0.0,
                   help="Deprecated/ignored. Training and inference use independent noise.")

    # Pseudo general-image regularization: copy one view V times and set all
    # camera extrinsics to identity. This prevents MVS adapters from overfitting
    # to multi-view synchronization at the cost of base FLUX visual quality.
    p.add_argument("--pseudo_general_prob", type=float, default=0.25,
                   help="Probability per micro-batch to use copied-view identity-camera regularization.")
    p.add_argument("--pseudo_general_random_view", action="store_true",
                   help="Randomly choose copied source view. Otherwise use anchor view 0.")

    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--hf_download", action="store_true")
    p.add_argument("--resume_mv_ckpt", default=None, help="Optional adapter checkpoint to continue training from.")

    # Periodic and end-of-training inference for visual evaluation.
    p.add_argument("--infer_every", type=int, default=500,
                   help="Run multi-view inference every N optimizer steps. Set <=0 to disable periodic inference.")
    p.add_argument("--no_infer_after_training", action="store_true")
    p.add_argument("--infer_manifest", default=None, help="Manifest used for sample generation. Defaults to train_manifest.")
    p.add_argument("--infer_sample_index", type=int, default=0)
    p.add_argument("--infer_num_steps", type=int, default=30)
    p.add_argument("--infer_seed", type=int, default=42)
    p.add_argument("--infer_guidance", type=float, default=None)
    p.add_argument("--infer_out", default=None, help="Final output image path. Defaults to output_dir/final_inference.jpg.")
    return p


def parse_args():
    # First pass: only discover --config.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()

    p = build_parser()
    config_data = {}
    if known.config:
        config_data = load_config_file(known.config)
        valid_keys = {a.dest for a in p._actions if a.dest != "help"}
        validate_config_keys(config_data, valid_keys, known.config)
        p.set_defaults(**config_data)

    args = p.parse_args()
    if args.train_manifest is None:
        p.error("--train_manifest is required, either in the config file or on the command line.")
    if args.output_dir is None:
        p.error("--output_dir is required, either in the config file or on the command line.")

    args.config_data = config_data
    return args

def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("flux_multiview_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_pseudo_general_regularization(
    pixel_values: torch.Tensor,
    cameras: torch.Tensor,
    probability: float,
    random_view: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Copy one view V times and set every camera to identity with probability p."""
    if probability <= 0.0 or torch.rand((), device=pixel_values.device).item() >= probability:
        return pixel_values, cameras, False

    b, v, c, h, w = pixel_values.shape
    if random_view:
        src_ids = torch.randint(0, v, (b,), device=pixel_values.device)
    else:
        src_ids = torch.zeros((b,), device=pixel_values.device, dtype=torch.long)

    batch_ids = torch.arange(b, device=pixel_values.device)
    src = pixel_values[batch_ids, src_ids]          # [B,3,H,W]
    pixel_values = src[:, None].repeat(1, v, 1, 1, 1).contiguous()

    identity = IDENTITY_CAMERA_12.to(device=cameras.device, dtype=cameras.dtype)
    cameras = identity[None, None, :].repeat(b, v, 1).contiguous()
    return pixel_values, cameras, True


def load_manifest_item(path: str | Path, index: int) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"sample_index={index} out of range for {path}")


def make_independent_noise(num_views: int, height: int, width: int, device, dtype, seed: int) -> torch.Tensor:
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
            draw.rectangle([0, 0, 130, 26], fill=(255, 255, 255))
            draw.text((6, 6), labels[i], fill=(0, 0, 0))
        imgs.append(img)
    w, h = imgs[0].size
    grid = Image.new("RGB", (w * len(imgs), h), color=(255, 255, 255))
    for i, img in enumerate(imgs):
        grid.paste(img, (i * w, 0))
    return grid


@torch.no_grad()
def run_sample_inference(args, model, ae, t5, clip, device, dtype, logger: logging.Logger) -> Path:
    manifest = args.infer_manifest or args.train_manifest
    out_path = Path(args.infer_out) if args.infer_out else Path(args.output_dir) / "final_inference.jpg"
    item = load_manifest_item(manifest, args.infer_sample_index)

    prompt = item.get("prompt", "a realistic 3D-rendered scene with a character performing an action")
    cameras = torch.tensor(item["extrinsics"][: args.num_views], dtype=torch.float32)
    cameras = cameras[None].to(device=device, dtype=dtype)

    guidance_value = args.guidance if args.infer_guidance is None else args.infer_guidance
    x = make_independent_noise(
        num_views=args.num_views,
        height=args.resolution,
        width=args.resolution,
        device=device,
        dtype=dtype,
        seed=args.infer_seed,
    )
    latent_h, latent_w = x.shape[-2], x.shape[-1]
    img = pack_latents(x)
    img_ids = make_img_ids(args.num_views, latent_h, latent_w, device, dtype)
    txt, txt_ids, y = encode_prompts(t5, clip, [prompt] * args.num_views, device, dtype)
    guidance = torch.full((args.num_views,), guidance_value, device=device, dtype=dtype)

    timesteps = get_schedule(args.infer_num_steps, img.shape[1], shift=(args.model_name != "flux-schnell"))
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

    labels = [f"view {i + 1}" for i in range(args.num_views)]
    grid = tensor_to_pil_grid(out, labels=labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)

    meta = {
        "manifest": str(manifest),
        "sample_index": args.infer_sample_index,
        "prompt": prompt,
        "num_views": args.num_views,
        "seed": args.infer_seed,
        "num_steps": args.infer_num_steps,
        "guidance": guidance_value,
        "output": str(out_path),
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Saved inference image: %s", out_path)
    return out_path


def run_periodic_inference(args, model, ae, t5, clip, device, dtype, logger: logging.Logger, writer: SummaryWriter, step: int) -> Path:
    infer_args = copy.copy(args)
    infer_args.infer_out = str(Path(args.output_dir) / "visualizations" / f"step_{step:06d}.jpg")
    path = run_sample_inference(infer_args, model, ae, t5, clip, device, dtype, logger)
    try:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).transpose(2, 0, 1)
        writer.add_image("infer/periodic_grid", arr, step)
    except Exception as exc:  # visualization logging must not break training
        logger.warning("Could not add periodic inference image to TensorBoard: %s", exc)
    return path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    seed_everything(args.seed)

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.mixed_precision]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(output_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    with open(output_dir / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    if getattr(args, "config_data", None):
        with open(output_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(args.config_data, f, indent=2, ensure_ascii=False)

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    writer.add_text("hyperparameters/json", json.dumps(vars(args), indent=2, ensure_ascii=False), 0)

    logger.info("Arguments: %s", json.dumps(vars(args), ensure_ascii=False))
    logger.info("Device=%s dtype=%s", device, dtype)
    if args.noise_share_ratio != 0.0:
        logger.info("--noise_share_ratio is deprecated and ignored. Independent noise is always used.")

    logger.info("Loading local text encoders and AE")
    t5, clip = load_local_text_encoders(device=device, dtype=dtype, max_length=512)

    try:
        ae = load_ae(args.model_name, device=device, hf_download=args.hf_download).eval().requires_grad_(False)
    except TypeError:
        ae = load_ae(args.model_name, device=device).eval().requires_grad_(False)

    logger.info("Loading FLUX MultiView transformer")
    model = load_multiview_flux(
        name=args.model_name,
        device=device,
        dtype=dtype,
        hf_download=args.hf_download,
        mv_adapter_dim=args.mv_adapter_dim,
        mv_dropout=args.mv_dropout,
        inject_single_blocks=args.inject_single_blocks,
        single_block_stride=args.single_block_stride,
        mv_attn_mode=args.mv_attn_mode,
        mv_use_timestep_modulation=not args.no_mv_timestep_modulation,
        mv_arch=args.mv_arch,
        mv_ckpt=args.resume_mv_ckpt,
    )
    model.freeze_base_model()
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()

    trainable, total = count_trainable_params(model)
    logger.info("Trainable params: %.2fM / total %.2fB", trainable / 1e6, total / 1e9)
    logger.info(
        "MVS config: arch=%s, attn_mode=%s, adapter_dim=%s, single_blocks=%s, "
        "single_block_stride=%s, timestep_modulation=%s, pseudo_general_prob=%.3f",
        args.mv_arch,
        args.mv_attn_mode,
        args.mv_adapter_dim,
        args.inject_single_blocks,
        args.single_block_stride,
        not args.no_mv_timestep_modulation,
        args.pseudo_general_prob,
    )
    writer.add_scalar("params/trainable", trainable, 0)
    writer.add_scalar("params/total", total, 0)

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
    logger.info("Dataset samples=%d manifest=%s", len(dataset), args.train_manifest)

    optimizer = torch.optim.AdamW(
        list(model.trainable_parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    micro_step = 0
    optim_step = 0
    running_loss = 0.0
    running_count = 0
    pseudo_count = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=args.max_steps)

    while optim_step < args.max_steps:
        for batch in loader:
            if optim_step >= args.max_steps:
                break

            pixel_values = batch["pixel_values"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            cameras = batch["cameras"].to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )
            prompts = batch["prompts"]

            pixel_values, cameras, used_pseudo = apply_pseudo_general_regularization(
                pixel_values=pixel_values,
                cameras=cameras,
                probability=args.pseudo_general_prob,
                random_view=args.pseudo_general_random_view,
            )
            pseudo_count += int(used_pseudo)

            b, v, c, h, w = pixel_values.shape
            assert v == args.num_views, f"Dataset returned V={v}, expected {args.num_views}"
            pixel_values = pixel_values.reshape(b * v, c, h, w)
            prompts_expanded = expand_prompts_for_views(prompts, v)

            with torch.no_grad():
                latents = ae.encode(pixel_values).to(dtype=dtype)

                # Required: independent Gaussian noise for every view.
                noise = torch.randn_like(latents)

                # All views of the same scene use the same timestep.
                t_scene = sample_flow_timesteps(b, device=device, dtype=dtype)
                t = t_scene.repeat_interleave(v)
                t_img = t.reshape(-1, 1, 1, 1)

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
            loss_raw = F.mse_loss(pred.float(), target.float())
            loss = loss_raw / args.grad_accum
            loss.backward()

            micro_step += 1
            running_loss += float(loss_raw.detach().cpu())
            running_count += 1
            writer.add_scalar("loss/micro_raw", float(loss_raw.detach().cpu()), micro_step)
            writer.add_scalar("batch/used_pseudo_general", int(used_pseudo), micro_step)

            if micro_step % args.grad_accum != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optim_step += 1
            pbar.update(1)

            writer.add_scalar("loss/step_raw", float(loss_raw.detach().cpu()), optim_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], optim_step)
            writer.add_scalar("train/grad_norm", float(grad_norm), optim_step)
            writer.add_scalar("train/pseudo_general_micro_batches", pseudo_count, optim_step)

            if optim_step % args.log_every == 0:
                avg = running_loss / max(1, running_count)
                logger.info(
                    "optim_step=%d micro_step=%d loss=%.6f pseudo_general_batches=%d",
                    optim_step,
                    micro_step,
                    avg,
                    pseudo_count,
                )
                writer.add_scalar("loss/running_avg", avg, optim_step)
                running_loss = 0.0
                running_count = 0

            if optim_step > 0 and optim_step % args.save_every == 0:
                save_path = output_dir / f"mv_adapter_step_{optim_step}.pt"
                torch.save(
                    {
                        "state_dict": extract_mv_state_dict(model),
                        "args": vars(args),
                        "global_step": optim_step,
                        "optim_step": optim_step,
                        "micro_step": micro_step,
                    },
                    save_path,
                )
                logger.info("Saved %s", save_path)

            if args.infer_every > 0 and optim_step > 0 and optim_step % args.infer_every == 0:
                logger.info("Running periodic inference at step %d", optim_step)
                model.eval()
                run_periodic_inference(args, model, ae, t5, clip, device, dtype, logger, writer, optim_step)
                model.train()

    final_path = output_dir / "mv_adapter_last.pt"
    torch.save(
        {
            "state_dict": extract_mv_state_dict(model),
            "args": vars(args),
            "global_step": optim_step,
            "optim_step": optim_step,
            "micro_step": micro_step,
        },
        final_path,
    )
    logger.info("Saved final adapter: %s", final_path)

    with open(output_dir / "train_state.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "optim_step": optim_step,
                "micro_step": micro_step,
                "pseudo_general_micro_batches": pseudo_count,
                "final_checkpoint": str(final_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not args.no_infer_after_training:
        logger.info("Running end-of-training inference with independent noise")
        model.eval()
        run_sample_inference(args, model, ae, t5, clip, device, dtype, logger)

    writer.flush()
    writer.close()
    pbar.close()


if __name__ == "__main__":
    main()
