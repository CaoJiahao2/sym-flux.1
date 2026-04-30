from __future__ import annotations

import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_sft

from flux.util import configs, print_load_warning

from .flux_multiview import FluxMultiView


def _first_existing_path(paths: list[str | os.PathLike | None]) -> str | None:
    for p in paths:
        if p is None:
            continue
        p = str(p)
        if p and os.path.isfile(p):
            return p
    return None


def resolve_flux_checkpoint(name: str = "flux-dev", hf_download: bool = False) -> str:
    """Resolve the FLUX transformer checkpoint path.

    Compatible with both older and newer black-forest-labs/flux util.py versions:
    - older ModelSpec may expose spec.ckpt_path;
    - newer ModelSpec may not, so we must use environment variables and repo_flow.

    Priority:
      1. FLUX_MODEL
      2. FLUX_DEV for flux-dev / FLUX_SCHNELL for flux-schnell
      3. spec.ckpt_path if present
      4. LOCAL_FLUX_DIR / spec.repo_flow
      5. /data/model_cjh/FLUX.1-dev / spec.repo_flow
      6. Hugging Face download only when hf_download=True
    """
    spec = configs[name]
    repo_flow = getattr(spec, "repo_flow", None)
    repo_id = getattr(spec, "repo_id", None)

    model_specific_env = None
    if name == "flux-dev":
        model_specific_env = os.getenv("FLUX_DEV")
    elif name == "flux-schnell":
        model_specific_env = os.getenv("FLUX_SCHNELL")

    candidates: list[str | os.PathLike | None] = [
        os.getenv("FLUX_MODEL"),
        model_specific_env,
        getattr(spec, "ckpt_path", None),
    ]

    local_dir = os.getenv("LOCAL_FLUX_DIR")
    if local_dir and repo_flow:
        candidates.append(Path(local_dir) / repo_flow)

    # Your current local path. Keeping this as a last local fallback is harmless.
    if repo_flow:
        candidates.append(Path("/data/model_cjh/FLUX.1-dev") / repo_flow)

    ckpt_path = _first_existing_path(candidates)
    if ckpt_path is not None:
        return ckpt_path

    if hf_download:
        if repo_id is None or repo_flow is None:
            raise RuntimeError(f"ModelSpec for {name} has no repo_id/repo_flow; cannot download checkpoint.")
        return hf_hub_download(repo_id, repo_flow)

    checked = "\n  - ".join(str(p) for p in candidates if p)
    raise FileNotFoundError(
        f"Cannot resolve FLUX checkpoint for {name}. Checked:\n  - {checked}\n"
        "Set FLUX_MODEL or FLUX_DEV to /data/model_cjh/FLUX.1-dev/flux1-dev.safetensors, "
        "or set HF_DOWNLOAD=1."
    )


def load_multiview_flux(
    name: str = "flux-dev",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    hf_download: bool = False,
    mv_adapter_dim: int = 512,
    mv_dropout: float = 0.0,
    inject_single_blocks: bool = False,
    mv_attn_mode: str = "same_token",
    mv_use_timestep_modulation: bool = True,
    mv_ckpt: str | None = None,
) -> FluxMultiView:
    spec = configs[name]
    model = FluxMultiView(
        params=spec.params,
        mv_adapter_dim=mv_adapter_dim,
        mv_dropout=mv_dropout,
        inject_single_blocks=inject_single_blocks,
        mv_attn_mode=mv_attn_mode,
        mv_use_timestep_modulation=mv_use_timestep_modulation,
    ).to(dtype=dtype)

    ckpt_path = resolve_flux_checkpoint(name, hf_download=hf_download)
    print(f"Loading base FLUX checkpoint: {ckpt_path}")
    sd = load_sft(ckpt_path, device=str(device))

    try:
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    except TypeError:
        # For older PyTorch without assign=True.
        missing, unexpected = model.load_state_dict(sd, strict=False)

    print_load_warning(missing, unexpected)

    if mv_ckpt:
        print(f"Loading multi-view adapter checkpoint: {mv_ckpt}")
        obj = torch.load(mv_ckpt, map_location="cpu")
        mv_sd = obj.get("state_dict", obj)
        missing, unexpected = model.load_state_dict(mv_sd, strict=False)
        print_load_warning(missing, unexpected)

    model.to(device)
    return model
