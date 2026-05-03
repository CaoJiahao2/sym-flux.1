from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
    """Resolve the FLUX transformer checkpoint path."""
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


def _load_sidecar_args(mv_ckpt: str | os.PathLike[str]) -> dict[str, Any]:
    """Load args.json from the checkpoint directory and fail hard if absent."""
    ckpt_path = Path(mv_ckpt)
    args_path = ckpt_path.parent / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(
            f"Required sidecar args file not found: {args_path}. "
            "For safety, MVS checkpoints must be loaded together with the args.json "
            "saved in the same output directory."
        )
    with args_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected {args_path} to contain a JSON object, got {type(obj).__name__}")
    return obj


def _same_bool(a: Any, b: Any) -> bool:
    return bool(a) == bool(b)


def _assert_architecture_matches_checkpoint(
    *,
    sidecar_args: dict[str, Any],
    name: str,
    mv_adapter_dim: int,
    mv_dropout: float,
    inject_single_blocks: bool,
    single_block_stride: int,
    mv_attn_mode: str,
    mv_use_timestep_modulation: bool,
    mv_arch: str,
    mv_ckpt: str | os.PathLike[str],
) -> None:
    """Ensure the runtime MVS architecture exactly matches the saved checkpoint.

    The source of truth is ``args.json`` in the checkpoint directory, not the
    checkpoint payload, because that file is written once at run start and is
    easier to inspect/version alongside training artifacts.
    """
    expected = {
        "model_name": name,
        "mv_arch": mv_arch,
        "mv_attn_mode": mv_attn_mode,
        "inject_single_blocks": inject_single_blocks,
        "single_block_stride": single_block_stride,
        "mv_dropout": mv_dropout,
        "no_mv_timestep_modulation": not mv_use_timestep_modulation,
    }
    if mv_arch == "adapter":
        expected["mv_adapter_dim"] = mv_adapter_dim

    mismatches: list[str] = []
    missing: list[str] = []
    for key, current_value in expected.items():
        if key not in sidecar_args:
            missing.append(key)
            continue
        saved_value = sidecar_args[key]
        if isinstance(current_value, bool):
            ok = _same_bool(saved_value, current_value)
        elif isinstance(current_value, float):
            try:
                ok = abs(float(saved_value) - current_value) <= 1e-12
            except (TypeError, ValueError):
                ok = False
        elif isinstance(current_value, int):
            try:
                ok = int(saved_value) == current_value
            except (TypeError, ValueError):
                ok = False
        else:
            ok = str(saved_value) == str(current_value)
        if not ok:
            mismatches.append(f"{key}: checkpoint args={saved_value!r}, runtime={current_value!r}")

    if missing or mismatches:
        details = []
        if missing:
            details.append("missing required key(s) in sidecar args.json: " + ", ".join(missing))
        if mismatches:
            details.append("architecture mismatch(s):\n  - " + "\n  - ".join(mismatches))
        raise ValueError(
            f"Refusing to load incompatible MVS checkpoint: {mv_ckpt}\n"
            + "\n".join(details)
        )


def _assert_mv_state_dict_exact(model: FluxMultiView, mv_sd: dict[str, torch.Tensor], mv_ckpt: str | os.PathLike[str]) -> None:
    """Require the saved adapter keys to match the current MVS module exactly."""
    expected_keys = {
        k for k in model.state_dict().keys()
        if k.startswith("mv_double_blocks") or k.startswith("mv_single_blocks")
    }
    saved_keys = set(mv_sd.keys())
    missing = sorted(expected_keys - saved_keys)
    unexpected = sorted(saved_keys - expected_keys)
    if missing or unexpected:
        def preview(keys: list[str], limit: int = 20) -> str:
            if not keys:
                return "none"
            suffix = "" if len(keys) <= limit else f" ... (+{len(keys) - limit} more)"
            return ", ".join(keys[:limit]) + suffix

        raise RuntimeError(
            f"MVS checkpoint key set does not exactly match current architecture: {mv_ckpt}\n"
            f"Missing MVS keys: {preview(missing)}\n"
            f"Unexpected MVS keys: {preview(unexpected)}"
        )


def load_multiview_flux(
    name: str = "flux-dev",
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    hf_download: bool = False,
    mv_adapter_dim: int = 512,
    mv_dropout: float = 0.0,
    inject_single_blocks: bool = True,
    single_block_stride: int = 4,
    mv_attn_mode: str = "full_view",
    mv_use_timestep_modulation: bool = True,
    mv_arch: str = "adapter",
    mv_ckpt: str | None = None,
) -> FluxMultiView:
    spec = configs[name]
    model = FluxMultiView(
        params=spec.params,
        mv_adapter_dim=mv_adapter_dim,
        mv_dropout=mv_dropout,
        inject_single_blocks=inject_single_blocks,
        single_block_stride=single_block_stride,
        mv_attn_mode=mv_attn_mode,
        mv_use_timestep_modulation=mv_use_timestep_modulation,
        mv_arch=mv_arch,
    ).to(dtype=dtype)

    ckpt_path = resolve_flux_checkpoint(name, hf_download=hf_download)
    print(f"Loading base FLUX checkpoint: {ckpt_path}")
    sd = load_sft(ckpt_path, device=str(device))

    try:
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    except TypeError:
        missing, unexpected = model.load_state_dict(sd, strict=False)

    print_load_warning(missing, unexpected)

    # For mv_arch='full_hidden', copy each FLUX double block img_attn into the
    # matching MVS view_attn before loading an optional trained MVS checkpoint.
    init_info = model.initialize_mv_attention_from_base()
    if init_info.get("double", 0) or init_info.get("single", 0):
        print(f"Initialized full-hidden MVS attention from base img_attn: {init_info}")

    if mv_ckpt:
        sidecar_args = _load_sidecar_args(mv_ckpt)
        _assert_architecture_matches_checkpoint(
            sidecar_args=sidecar_args,
            name=name,
            mv_adapter_dim=mv_adapter_dim,
            mv_dropout=mv_dropout,
            inject_single_blocks=inject_single_blocks,
            single_block_stride=single_block_stride,
            mv_attn_mode=mv_attn_mode,
            mv_use_timestep_modulation=mv_use_timestep_modulation,
            mv_arch=mv_arch,
            mv_ckpt=mv_ckpt,
        )

        print(f"Loading multi-view adapter checkpoint: {mv_ckpt}")
        obj = torch.load(mv_ckpt, map_location="cpu")
        mv_sd = obj.get("state_dict", obj)
        if not isinstance(mv_sd, dict):
            raise TypeError(f"Checkpoint state_dict must be a dict, got {type(mv_sd).__name__}: {mv_ckpt}")
        _assert_mv_state_dict_exact(model, mv_sd, mv_ckpt)
        # strict=False is used only because mv_sd intentionally contains MVS keys
        # and omits all frozen base FLUX keys. The MVS key set was already checked
        # exactly above, so this will still fail on tensor-shape incompatibilities.
        missing, unexpected = model.load_state_dict(mv_sd, strict=False)
        mv_missing = [k for k in missing if k.startswith("mv_double_blocks") or k.startswith("mv_single_blocks")]
        mv_unexpected = [k for k in unexpected if k.startswith("mv_double_blocks") or k.startswith("mv_single_blocks")]
        if mv_missing or mv_unexpected:
            raise RuntimeError(
                f"Unexpected MVS load result for {mv_ckpt}: missing={mv_missing}, unexpected={mv_unexpected}"
            )

    model.to(device)
    return model
