from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel, CLIPTextModel


class LocalT5Embedder(nn.Module):
    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_dir: str | Path,
        max_length: int = 512,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.model_dir = str(model_dir)
        self.tokenizer_dir = str(tokenizer_dir)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            local_files_only=True,
        )
        self.model = T5EncoderModel.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device

        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )
        return out.last_hidden_state


class LocalCLIPEmbedder(nn.Module):
    def __init__(
        self,
        model_dir: str | Path,
        tokenizer_dir: str | Path,
        max_length: int = 77,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.model_dir = str(model_dir)
        self.tokenizer_dir = str(tokenizer_dir)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            local_files_only=True,
        )
        self.model = CLIPTextModel.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device

        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output

        return out.last_hidden_state[:, 0]


def load_local_text_encoders(
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 512,
):
    local_flux_dir = Path(os.environ.get("LOCAL_FLUX_DIR", "/data/model_cjh/FLUX.1-dev"))

    t5_model_dir = local_flux_dir / "text_encoder_2"
    t5_tokenizer_dir = local_flux_dir / "tokenizer_2"

    clip_model_dir = local_flux_dir / "text_encoder"
    clip_tokenizer_dir = local_flux_dir / "tokenizer"

    for p in [t5_model_dir, t5_tokenizer_dir, clip_model_dir, clip_tokenizer_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Local text encoder path not found: {p}")

    print(f"[LOCAL] loading T5 from: {t5_model_dir}")
    t5 = LocalT5Embedder(
        model_dir=t5_model_dir,
        tokenizer_dir=t5_tokenizer_dir,
        max_length=max_length,
        torch_dtype=dtype,
    ).to(device)

    print(f"[LOCAL] loading CLIP from: {clip_model_dir}")
    clip = LocalCLIPEmbedder(
        model_dir=clip_model_dir,
        tokenizer_dir=clip_tokenizer_dir,
        max_length=77,
        torch_dtype=dtype,
    ).to(device)

    t5.eval().requires_grad_(False)
    clip.eval().requires_grad_(False)

    return t5, clip
