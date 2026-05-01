from __future__ import annotations

import torch
from torch import Tensor, nn


class MultiViewSyncBlock(nn.Module):
    """SynCamMaster-style cross-view synchronization for FLUX image tokens.

    Input image tokens are arranged as [B * V, S, D]. The block reshapes them to
    [B, V, S, D], adds a camera embedding per view, performs cross-view attention,
    and returns a residual update.

    Two attention modes are supported:
        - same_token: [B*S, V, A]. Cheap; every spatial token attends only to the
          token with the same spatial index in other views.
        - full_view: [B, V*S, A]. Closer to SynCamMaster; tokens attend to all
          spatial tokens from all views. More expensive.

    The timestep/guidance modulation is deliberately lightweight. Instead of a
    full per-block hidden -> 3*hidden modulation MLP, which is very large for
    FLUX.1-dev, this block reuses the frozen FLUX block modulation outputs passed
    from FluxMultiView and only learns a tiny [shift, scale, gate] bias.
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
            raise ValueError(f"Unsupported attn_mode={attn_mode}. Use 'same_token' or 'full_view'.")

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
        self.attn_mode = attn_mode
        self.use_timestep_modulation = use_timestep_modulation

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.down = nn.Linear(hidden_size, adapter_dim, bias=True)

        # Do not LayerNorm the 12 camera values. Rotation/translation scale should
        # be normalized in camera preprocessing, not erased here.
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

        if use_timestep_modulation:
            # SynCamMaster has per-block modulation_mvs initialized from the base
            # modulation. For FLUX we reuse the frozen base modulation and learn a
            # tiny additive bias, avoiding a huge per-block MLP.
            self.modulation_bias = nn.Parameter(torch.zeros(1, 3, hidden_size))
        else:
            self.register_parameter("modulation_bias", None)

        # Non-destructive insertion: at step 0 this block is an exact residual no-op.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        img_tokens: Tensor,
        cameras: Tensor,
        num_views: int,
        mod_shift: Tensor | None = None,
        mod_scale: Tensor | None = None,
        mod_gate: Tensor | None = None,
    ) -> Tensor:
        """Apply cross-view synchronization.

        Args:
            img_tokens: [B*V, S, D].
            cameras: [B, V, 12], relative [R|t] extrinsics.
            num_views: V.
            mod_shift/mod_scale/mod_gate: optional frozen FLUX modulation outputs,
                each with shape [B*V, 1, D].
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

        if self.use_timestep_modulation:
            if mod_shift is None or mod_scale is None or mod_gate is None:
                raise ValueError(
                    "mod_shift/mod_scale/mod_gate must be provided when "
                    "use_timestep_modulation=True."
                )
            bias_shift, bias_scale, bias_gate = self.modulation_bias.to(
                dtype=x_norm.dtype,
                device=x_norm.device,
            ).chunk(3, dim=1)
            shift = mod_shift.to(dtype=x_norm.dtype, device=x_norm.device) + bias_shift
            scale = mod_scale.to(dtype=x_norm.dtype, device=x_norm.device) + bias_scale
            gate = mod_gate.to(dtype=x_norm.dtype, device=x_norm.device) + bias_gate
            x_norm = (1.0 + scale) * x_norm + shift

        # reshape instead of view: single-block image slices are often non-contiguous.
        x = x_norm.reshape(batch_size, num_views, seq_len, dim)  # [B,V,S,D]
        x_small = self.down(x)                                  # [B,V,S,A]

        cam = self.camera_encoder(cameras.to(dtype=x_small.dtype, device=x_small.device))
        x_small = x_small + cam[:, :, None, :]                  # [B,V,S,A]

        if self.attn_mode == "same_token":
            # [B,V,S,A] -> [B*S,V,A]
            y = x_small.permute(0, 2, 1, 3).reshape(
                batch_size * seq_len,
                num_views,
                self.adapter_dim,
            )
            y, _ = self.view_attn(y, y, y, need_weights=False)
            y = y.reshape(
                batch_size,
                seq_len,
                num_views,
                self.adapter_dim,
            ).permute(0, 2, 1, 3)  # [B,V,S,A]
        else:
            # [B,V,S,A] -> [B,V*S,A], closer to SynCamMaster.
            y = x_small.reshape(batch_size, num_views * seq_len, self.adapter_dim)
            y, _ = self.view_attn(y, y, y, need_weights=False)
            y = y.reshape(batch_size, num_views, seq_len, self.adapter_dim)

        update = self.up(y).reshape(bv, seq_len, dim)
        if gate is not None:
            update = gate.to(dtype=update.dtype, device=update.device) * update
        return img_tokens + update
