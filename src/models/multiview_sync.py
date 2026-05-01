from __future__ import annotations

import torch
from torch import Tensor, nn

from flux.modules.layers import SelfAttention


class MultiViewSyncBlock(nn.Module):
    """SynCamMaster-style cross-view synchronization for FLUX image tokens.

    Input image tokens are arranged as [B * V, S, D]. The block reshapes them to
    [B, V, S, D], adds explicit camera + spatial/view positional embeddings,
    performs cross-view attention, and returns a residual update.

    Attention modes:
        - same_token: [B*S, V, A/D]. Cheap; every spatial token attends only to
          the token with the same spatial index in other views.
        - full_view: [B, V*S, A/D]. Closer to SynCamMaster; tokens attend to all
          spatial tokens from all views. This is now the default configuration.

    Architectures:
        - adapter: low-dimensional adapter. Hidden D -> A -> MHA(A) -> D.
        - full_hidden: hidden-size view attention. Uses FLUX SelfAttention at D
          and can be initialized from each double block's img_attn weights.

    The residual projector is zero-initialized, so insertion is non-destructive
    at step 0. Camera and position encoders are also initialized conservatively.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        adapter_dim: int = 512,
        dropout: float = 0.0,
        attn_mode: str = "full_view",
        use_timestep_modulation: bool = True,
        mv_arch: str = "adapter",
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if attn_mode not in {"same_token", "full_view"}:
            raise ValueError(f"Unsupported attn_mode={attn_mode}. Use 'same_token' or 'full_view'.")
        if mv_arch not in {"adapter", "full_hidden"}:
            raise ValueError(f"Unsupported mv_arch={mv_arch}. Use 'adapter' or 'full_hidden'.")

        self.hidden_size = hidden_size
        self.mv_arch = mv_arch
        self.attn_mode = attn_mode
        self.use_timestep_modulation = use_timestep_modulation
        self.is_full_hidden = mv_arch == "full_hidden"
        self.attn_dim = hidden_size if self.is_full_hidden else adapter_dim

        if self.is_full_hidden:
            self.num_adapter_heads = num_heads
        elif self.attn_dim % num_heads != 0:
            # Use a valid number of adapter heads when adapter_dim is small.
            valid_heads = max(1, min(num_heads, self.attn_dim // 64))
            while self.attn_dim % valid_heads != 0 and valid_heads > 1:
                valid_heads -= 1
            self.num_adapter_heads = valid_heads
        else:
            self.num_adapter_heads = num_heads

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if self.is_full_hidden:
            self.down = nn.Identity()
            self.view_attn = SelfAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
            )
            self.projector = nn.Linear(hidden_size, hidden_size, bias=True)
        else:
            self.down = nn.Linear(hidden_size, self.attn_dim, bias=True)
            self.view_attn = nn.MultiheadAttention(
                embed_dim=self.attn_dim,
                num_heads=self.num_adapter_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.projector = nn.Linear(self.attn_dim, hidden_size, bias=True)

        # Do not LayerNorm the 12 camera values. Rotation/translation scale should
        # be normalized in camera preprocessing, not erased here.
        self.camera_encoder = nn.Sequential(
            nn.Linear(12, self.attn_dim),
            nn.SiLU(),
            nn.Linear(self.attn_dim, self.attn_dim),
        )

        # Explicit view/spatial positional encoding. Input = [view_id, y, x]
        # normalized to roughly [0, 1]. This is important for full_view mode,
        # where V*S tokens are mixed in a single sequence.
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.attn_dim),
            nn.SiLU(),
            nn.Linear(self.attn_dim, self.attn_dim),
        )

        if use_timestep_modulation:
            # Reuse frozen FLUX block modulation and learn a tiny additive bias.
            self.modulation_bias = nn.Parameter(torch.zeros(1, 3, hidden_size))
        else:
            self.register_parameter("modulation_bias", None)

        # Non-destructive insertion: at step 0 this block is an exact residual no-op.
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)

        # Conservative start for camera and position branches.
        if isinstance(self.camera_encoder[-1], nn.Linear):
            nn.init.zeros_(self.camera_encoder[-1].weight)
            nn.init.zeros_(self.camera_encoder[-1].bias)
        if isinstance(self.position_encoder[-1], nn.Linear):
            nn.init.zeros_(self.position_encoder[-1].weight)
            nn.init.zeros_(self.position_encoder[-1].bias)

    @torch.no_grad()
    def init_from_flux_img_attn(self, img_attn: nn.Module) -> bool:
        """Initialize full-hidden view attention from FLUX image self-attention.

        Returns True if weights were copied. For low-dimensional adapter mode this
        is a no-op because the dimensions do not match.
        """
        if not self.is_full_hidden:
            return False
        self.view_attn.load_state_dict(img_attn.state_dict(), strict=True)
        return True

    def _position_features(
        self,
        img_ids: Tensor | None,
        batch_size: int,
        num_views: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Return [B,V,S,3] with explicit [view_id, normalized_y, normalized_x]."""
        view_id = torch.arange(num_views, device=device, dtype=dtype)[None, :, None, None]
        view_id = view_id.expand(batch_size, num_views, seq_len, 1)
        view_id = view_id / max(1, num_views - 1)

        if img_ids is None:
            # Fallback: infer a square-ish raster order when ids are unavailable.
            side = int(seq_len**0.5)
            if side * side == seq_len:
                yy = torch.arange(side, device=device, dtype=dtype)[:, None].expand(side, side)
                xx = torch.arange(side, device=device, dtype=dtype)[None, :].expand(side, side)
                xy = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
            else:
                idx = torch.arange(seq_len, device=device, dtype=dtype)
                xy = torch.stack([idx, torch.zeros_like(idx)], dim=-1)
            xy = xy[None, None].expand(batch_size, num_views, seq_len, 2)
        else:
            ids = img_ids.reshape(batch_size, num_views, seq_len, 3).to(device=device, dtype=dtype)
            xy = ids[..., 1:3]

        denom = xy.amax(dim=(1, 2), keepdim=True).clamp_min(1.0)
        xy = xy / denom
        return torch.cat([view_id, xy], dim=-1)

    def _full_view_ids(
        self,
        img_ids: Tensor,
        batch_size: int,
        num_views: int,
        seq_len: int,
    ) -> Tensor:
        ids = img_ids.reshape(batch_size, num_views, seq_len, 3).clone()
        view_ids = torch.arange(num_views, device=ids.device, dtype=ids.dtype)
        ids[..., 0] = view_ids[None, :, None]
        return ids.reshape(batch_size, num_views * seq_len, 3)

    def _same_token_ids(
        self,
        img_ids: Tensor,
        batch_size: int,
        num_views: int,
        seq_len: int,
    ) -> Tensor:
        ids = img_ids.reshape(batch_size, num_views, seq_len, 3).clone()
        ids = ids.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_views, 3)
        view_ids = torch.arange(num_views, device=ids.device, dtype=ids.dtype)
        ids[..., 0] = view_ids[None, :]
        return ids

    def forward(
        self,
        img_tokens: Tensor,
        cameras: Tensor,
        num_views: int,
        img_ids: Tensor | None = None,
        pe_embedder: nn.Module | None = None,
        mod_shift: Tensor | None = None,
        mod_scale: Tensor | None = None,
        mod_gate: Tensor | None = None,
    ) -> Tensor:
        """Apply cross-view synchronization.

        Args:
            img_tokens: [B*V, S, D].
            cameras: [B, V, 12], relative [R|t] extrinsics.
            num_views: V.
            img_ids: [B*V, S, 3], FLUX spatial ids.
            pe_embedder: FLUX EmbedND. Required for full_hidden mode.
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

        x = x_norm.reshape(batch_size, num_views, seq_len, dim)  # [B,V,S,D]
        x_sync = self.down(x)                                    # [B,V,S,A/D]

        cam = self.camera_encoder(cameras.to(dtype=x_sync.dtype, device=x_sync.device))
        x_sync = x_sync + cam[:, :, None, :]

        pos_feat = self._position_features(
            img_ids=img_ids,
            batch_size=batch_size,
            num_views=num_views,
            seq_len=seq_len,
            dtype=x_sync.dtype,
            device=x_sync.device,
        )
        x_sync = x_sync + self.position_encoder(pos_feat)

        if self.attn_mode == "same_token":
            # [B,V,S,A/D] -> [B*S,V,A/D]
            y = x_sync.permute(0, 2, 1, 3).reshape(batch_size * seq_len, num_views, self.attn_dim)
            if self.is_full_hidden:
                if pe_embedder is None or img_ids is None:
                    raise ValueError("full_hidden MVS requires img_ids and pe_embedder.")
                attn_ids = self._same_token_ids(img_ids, batch_size, num_views, seq_len)
                pe = pe_embedder(attn_ids)
                y = self.view_attn(y, pe=pe)
            else:
                y, _ = self.view_attn(y, y, y, need_weights=False)
            y = y.reshape(batch_size, seq_len, num_views, self.attn_dim).permute(0, 2, 1, 3)
        else:
            # [B,V,S,A/D] -> [B,V*S,A/D], closer to SynCamMaster.
            y = x_sync.reshape(batch_size, num_views * seq_len, self.attn_dim)
            if self.is_full_hidden:
                if pe_embedder is None or img_ids is None:
                    raise ValueError("full_hidden MVS requires img_ids and pe_embedder.")
                attn_ids = self._full_view_ids(img_ids, batch_size, num_views, seq_len)
                pe = pe_embedder(attn_ids)
                y = self.view_attn(y, pe=pe)
            else:
                y, _ = self.view_attn(y, y, y, need_weights=False)
            y = y.reshape(batch_size, num_views, seq_len, self.attn_dim)

        update = self.projector(y).reshape(bv, seq_len, dim)
        if gate is not None:
            update = gate.to(dtype=update.dtype, device=update.device) * update
        return img_tokens + update
