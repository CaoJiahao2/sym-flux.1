from __future__ import annotations

import torch
from torch import Tensor, nn

from flux.modules.layers import Modulation


class MultiViewSyncBlock(nn.Module):
    """SynCamMaster-style cross-view synchronization for FLUX image tokens.

    Input:
        img_tokens: [B*V, S, D]
        cameras:    [B, V, 12]
        vec:        [B*V, D], FLUX timestep/text guidance conditioning vector

    attn_mode:
        - same_token: [B*S, V, A], cheap; good for stage-1.
        - full_view:  [B, V*S, A], closer to SynCamMaster; better for larger view gaps.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        adapter_dim: int = 512,
        dropout: float = 0.0,
        attn_mode: str = "same_token",
        use_timestep_modulation: bool = True,
    ) -> None:
        super().__init__()

        if attn_mode not in {"same_token", "full_view"}:
            raise ValueError(f"Unsupported attn_mode={attn_mode}")

        if adapter_dim % num_heads != 0:
            valid_heads = max(1, min(num_heads, adapter_dim // 64))
            while adapter_dim % valid_heads != 0 and valid_heads > 1:
                valid_heads -= 1
            num_adapter_heads = valid_heads
        else:
            num_adapter_heads = num_heads

        self.hidden_size = hidden_size
        self.adapter_dim = adapter_dim
        self.num_adapter_heads = num_adapter_heads
        self.attn_mode = attn_mode
        self.use_timestep_modulation = use_timestep_modulation

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Use FLUX/SynCamMaster-style timestep-conditioned AdaLN gate.
        # SynCamMaster-style lightweight modulation offset.
        # Only 3 * hidden_size parameters per block, not hidden_size * 3hidden_size.
        if use_timestep_modulation:
            self.modulation_mvs = nn.Parameter(
                torch.zeros(1, 3, hidden_size)
            )
        else:
            self.modulation_mvs = None

        self.down = nn.Linear(hidden_size, adapter_dim, bias=True)

        # Important: do NOT LayerNorm the 12 pose values.
        # Pose scale should be handled in camera preprocessing, not normalized away here.
        self.camera_encoder = nn.Sequential(
            nn.Linear(12, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, adapter_dim),
        )

        self.view_attn = nn.MultiheadAttention(
            embed_dim=adapter_dim,
            num_heads=num_adapter_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.up = nn.Linear(adapter_dim, hidden_size, bias=True)

        # Non-destructive insertion: at initialization this block is exact no-op.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        img_tokens: Tensor,
        cameras: Tensor,
        num_views: int,
        vec: Tensor | None = None,
    ) -> Tensor:
        """Apply cross-view synchronization.

        Args:
            img_tokens: [B*V, S, D]
            cameras:    [B, V, 12]
            num_views:  V
            vec:        [B*V, D], same vector used by FLUX blocks.
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

        x_norm = self.norm(img_tokens)

        gate = None
        if self.modulation is not None:
            if vec is None:
                raise ValueError("vec must be provided when use_timestep_modulation=True")
            if vec.shape[0] != bv:
                raise ValueError(f"Expected vec batch {bv}, got {vec.shape[0]}")

            mod, _ = self.modulation(vec.to(dtype=x_norm.dtype, device=x_norm.device))
            x_norm = (1.0 + mod.scale) * x_norm + mod.shift
            gate = mod.gate

        # reshape instead of view, because single-block slicing may be non-contiguous.
        x = x_norm.reshape(batch_size, num_views, seq_len, dim)  # [B,V,S,D]
        x_small = self.down(x)                                  # [B,V,S,A]

        cam = self.camera_encoder(
            cameras.to(dtype=x_small.dtype, device=x_small.device)
        )                                                       # [B,V,A]
        x_small = x_small + cam[:, :, None, :]                  # [B,V,S,A]

        if self.attn_mode == "same_token":
            # Cheap version: each spatial token attends only to corresponding
            # spatial tokens from other views.
            y = x_small.permute(0, 2, 1, 3).reshape(
                batch_size * seq_len,
                num_views,
                self.adapter_dim,
            )                                                   # [B*S,V,A]
            y, _ = self.view_attn(y, y, y, need_weights=False)
            y = y.reshape(
                batch_size,
                seq_len,
                num_views,
                self.adapter_dim,
            ).permute(0, 2, 1, 3)                               # [B,V,S,A]

        else:
            # Closer to SynCamMaster:
            # one view token can attend to all spatial tokens from all views.
            y = x_small.reshape(
                batch_size,
                num_views * seq_len,
                self.adapter_dim,
            )                                                   # [B,V*S,A]
            y, _ = self.view_attn(y, y, y, need_weights=False)
            y = y.reshape(
                batch_size,
                num_views,
                seq_len,
                self.adapter_dim,
            )                                                   # [B,V,S,A]

        update = self.up(y).reshape(bv, seq_len, dim)

        if gate is not None:
            update = gate.to(dtype=update.dtype) * update

        return img_tokens + update