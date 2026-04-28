from __future__ import annotations

import math
from typing import Iterable

import torch
from einops import rearrange, repeat
from torch import Tensor


def pack_latents(x: Tensor) -> Tensor:
    """Pack AE latents [B,16,H,W] into FLUX img tokens [B,H/2*W/2,64]."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")
    if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
        raise ValueError(f"Latent H/W must be even for FLUX 2x2 packing, got {tuple(x.shape)}")
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpack_latents(x: Tensor, latent_h: int, latent_w: int) -> Tensor:
    """Unpack FLUX img tokens [B,H/2*W/2,64] into AE latents [B,16,H,W]."""
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=latent_h // 2,
        w=latent_w // 2,
        ph=2,
        pw=2,
    )


def make_img_ids(batch_size: int, latent_h: int, latent_w: int, device, dtype) -> Tensor:
    """Create FLUX image positional ids after 2x2 latent packing."""
    h = latent_h // 2
    w = latent_w // 2
    img_ids = torch.zeros((h, w, 3), device=device, dtype=dtype)
    img_ids[..., 1] = torch.arange(h, device=device, dtype=dtype)[:, None]
    img_ids[..., 2] = torch.arange(w, device=device, dtype=dtype)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids


@torch.no_grad()
def encode_prompts(t5, clip, prompts: list[str], device, dtype) -> tuple[Tensor, Tensor, Tensor]:
    """Encode text into FLUX txt, txt_ids and vector y.

    Official BFL flux text embedders are callable modules. T5 returns token
    embeddings [B,L,4096], CLIP returns pooled vectors [B,768].
    """
    txt = t5(prompts).to(device=device, dtype=dtype)
    y = clip(prompts).to(device=device, dtype=dtype)
    txt_ids = torch.zeros((txt.shape[0], txt.shape[1], 3), device=device, dtype=dtype)
    return txt, txt_ids, y


def expand_prompts_for_views(prompts: list[str], num_views: int) -> list[str]:
    out: list[str] = []
    for prompt in prompts:
        out.extend([prompt] * num_views)
    return out


def sample_flow_timesteps(batch_size: int, device, dtype) -> Tensor:
    """Logit-normal timestep sampling used commonly for rectified-flow training."""
    return torch.sigmoid(torch.randn((batch_size,), device=device, dtype=dtype))


def shared_view_noise_like(x: Tensor, num_views: int, share_ratio: float = 0.75) -> Tensor:
    """Create partially shared noise for B*V latents.

    During training, completely independent noise can weaken cross-view coupling.
    This function mixes a shared per-scene noise component with independent
    per-view noise. Set share_ratio=0.0 for independent noise.
    """
    if share_ratio <= 0:
        return torch.randn_like(x)
    bv = x.shape[0]
    if bv % num_views != 0:
        raise ValueError(f"B*V={bv} not divisible by num_views={num_views}")
    b = bv // num_views
    base = torch.randn((b, 1, *x.shape[1:]), device=x.device, dtype=x.dtype)
    indep = torch.randn((b, num_views, *x.shape[1:]), device=x.device, dtype=x.dtype)
    noise = share_ratio * base + math.sqrt(max(1e-8, 1.0 - share_ratio**2)) * indep
    return noise.reshape_as(x)


def count_trainable_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
