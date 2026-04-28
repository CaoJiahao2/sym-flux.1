from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)


class LocalT5Embedder(nn.Module):
    """BFL-FLUX compatible local T5 embedder.

    Official flux.util.load_t5 returns a module whose forward(prompts) returns
    T5 hidden states [B, L, 4096]. This class keeps the same interface but loads
    from /path/to/FLUX.1-dev/text_encoder_2 and tokenizer_2 with local_files_only.
    """

    def __init__(self, local_flux_dir: str | Path, max_length: int = 512, torch_dtype=torch.bfloat16):
        super().__init__()
        local_flux_dir = Path(local_flux_dir)
        model_dir = local_flux_dir / "text_encoder_2"
        tokenizer_dir = local_flux_dir / "tokenizer_2"
        if not model_dir.exists():
            raise FileNotFoundError(f"T5 text_encoder_2 directory not found: {model_dir}")
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"T5 tokenizer_2 directory not found: {tokenizer_dir}")
        print(f"[TEXT] Loading local T5 from {model_dir}")
        print(f"[TEXT] Loading local T5 tokenizer from {tokenizer_dir}")
        self.tokenizer = T5Tokenizer.from_pretrained(
            str(tokenizer_dir),
            max_length=max_length,
            local_files_only=True,
        )
        self.model = T5EncoderModel.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, prompts: Sequence[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            list(prompts),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


class LocalCLIPEmbedder(nn.Module):
    """BFL-FLUX compatible local CLIP embedder.

    Official flux.util.load_clip returns pooled CLIP vector [B, 768].
    """

    def __init__(self, local_flux_dir: str | Path, torch_dtype=torch.bfloat16):
        super().__init__()
        local_flux_dir = Path(local_flux_dir)
        model_dir = local_flux_dir / "text_encoder"
        tokenizer_dir = local_flux_dir / "tokenizer"
        if not model_dir.exists():
            raise FileNotFoundError(f"CLIP text_encoder directory not found: {model_dir}")
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"CLIP tokenizer directory not found: {tokenizer_dir}")
        print(f"[TEXT] Loading local CLIP from {model_dir}")
        print(f"[TEXT] Loading local CLIP tokenizer from {tokenizer_dir}")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            str(tokenizer_dir),
            local_files_only=True,
        )
        self.model = CLIPTextModel.from_pretrained(
            str(model_dir),
            torch_dtype=torch_dtype,
            local_files_only=True,
        )

    @torch.no_grad()
    def forward(self, prompts: Sequence[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = self.tokenizer(
            list(prompts),
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]


def load_local_t5(local_flux_dir: str | Path, device, max_length: int = 512, torch_dtype=torch.bfloat16):
    return LocalT5Embedder(local_flux_dir, max_length=max_length, torch_dtype=torch_dtype).to(device)


def load_local_clip(local_flux_dir: str | Path, device, torch_dtype=torch.bfloat16):
    return LocalCLIPEmbedder(local_flux_dir, torch_dtype=torch_dtype).to(device)
