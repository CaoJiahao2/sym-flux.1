from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiViewSyncBlock(nn.Module):
    """SynCamMaster-style cross-view synchronization for FLUX image tokens.

    Input image tokens are arranged as [B * V, S, D]. The block reshapes them to
    [B, V, S, D], adds a camera embedding per view, performs self-attention over
    the V views for each spatial token, and returns a residual update.

    The attention itself is performed in a lower adapter dimension to keep the
    number of trainable parameters practical for FLUX.1-dev.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        adapter_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if adapter_dim % num_heads != 0:
            # Use a valid number of adapter heads even when adapter_dim is small.
            valid_heads = max(1, min(num_heads, adapter_dim // 64))
            while adapter_dim % valid_heads != 0 and valid_heads > 1:
                valid_heads -= 1
            num_adapter_heads = valid_heads
        else:
            num_adapter_heads = num_heads

        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.num_adapter_heads = num_adapter_heads

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.down = nn.Linear(hidden_size, adapter_dim, bias=True)
        self.camera_encoder = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, adapter_dim),
        )
        self.view_attn = nn.MultiheadAttention(
            adapter_dim,
            num_adapter_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.up = nn.Linear(adapter_dim, hidden_size, bias=True)

        # Critical for non-destructive insertion: at step 0 this block is an
        # exact residual no-op, so loading the original FLUX checkpoint remains
        # behavior-preserving before training.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, img_tokens: Tensor, cameras: Tensor, num_views: int) -> Tensor:
        """Apply cross-view attention.

        Args:
            img_tokens: Tensor [B*V, S, D].
            cameras: Tensor [B, V, 12], relative [R|t] extrinsics.
            num_views: V.
        """
        if cameras is None or num_views <= 1:
            return img_tokens

        bv, seq_len, dim = img_tokens.shape
        if bv % num_views != 0:
            raise ValueError(f"B*V={bv} is not divisible by num_views={num_views}")
        batch_size = bv // num_views

        if cameras.shape[:2] != (batch_size, num_views) or cameras.shape[-1] != 12:
            raise ValueError(
                f"Expected cameras [B,V,12] = [{batch_size},{num_views},12], "
                f"got {tuple(cameras.shape)}"
            )

        x = img_tokens.view(batch_size, num_views, seq_len, dim)  # [B,V,S,D]
        x_small = self.down(self.norm(x))                         # [B,V,S,A]
        cam = self.camera_encoder(cameras.to(dtype=x_small.dtype, device=x_small.device))
        x_small = x_small + cam[:, :, None, :]                    # [B,V,S,A]

        # Attention over views for every spatial token independently.
        y = x_small.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_views, self.adapter_dim)
        y, _ = self.view_attn(y, y, y, need_weights=False)
        y = y.view(batch_size, seq_len, num_views, self.adapter_dim).permute(0, 2, 1, 3)

        update = self.up(y).reshape(bv, seq_len, dim)
        return img_tokens + update
