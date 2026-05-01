from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from flux.model import Flux, FluxParams
from flux.modules.layers import timestep_embedding

from .multiview_sync import MultiViewSyncBlock


class FluxMultiView(Flux):
    """Non-destructive FLUX.1 multi-view extension.

    This subclass leaves the official black-forest-labs/flux source unchanged.
    It reuses the original Flux modules and inserts trainable multi-view sync
    adapters after double-stream blocks and optionally after selected
    single-stream blocks.

    Load original FLUX weights with strict=False. Missing keys should be only
    mv_double_blocks / mv_single_blocks.
    """

    def __init__(
        self,
        params: FluxParams,
        mv_adapter_dim: int = 512,
        mv_dropout: float = 0.0,
        inject_single_blocks: bool = False,
        single_block_stride: int = 4,
        mv_attn_mode: str = "same_token",
        mv_use_timestep_modulation: bool = True,
    ) -> None:
        super().__init__(params)
        if single_block_stride <= 0:
            raise ValueError("single_block_stride must be a positive integer")

        self.inject_single_blocks = inject_single_blocks
        self.single_block_stride = single_block_stride
        self.mv_attn_mode = mv_attn_mode
        self.mv_use_timestep_modulation = mv_use_timestep_modulation
        self._gradient_checkpointing = False

        self.mv_double_blocks = nn.ModuleList([
            MultiViewSyncBlock(
                hidden_size=params.hidden_size,
                num_heads=params.num_heads,
                adapter_dim=mv_adapter_dim,
                dropout=mv_dropout,
                attn_mode=mv_attn_mode,
                use_timestep_modulation=mv_use_timestep_modulation,
            )
            for _ in range(params.depth)
        ])

        if inject_single_blocks:
            self.single_block_indices = list(range(0, params.depth_single_blocks, single_block_stride))
            self.mv_single_blocks = nn.ModuleDict({
                str(i): MultiViewSyncBlock(
                    hidden_size=params.hidden_size,
                    num_heads=params.num_heads,
                    adapter_dim=mv_adapter_dim,
                    dropout=mv_dropout,
                    attn_mode=mv_attn_mode,
                    use_timestep_modulation=mv_use_timestep_modulation,
                )
                for i in self.single_block_indices
            })
        else:
            self.single_block_indices = []
            self.mv_single_blocks = nn.ModuleDict()

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False

    def freeze_base_model(self) -> None:
        """Freeze all official FLUX parameters; keep only MVS adapters trainable."""
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.mv_double_blocks.parameters():
            p.requires_grad_(True)
        for p in self.mv_single_blocks.parameters():
            p.requires_grad_(True)

    def trainable_parameters(self):
        for p in self.mv_double_blocks.parameters():
            if p.requires_grad:
                yield p
        for p in self.mv_single_blocks.parameters():
            if p.requires_grad:
                yield p

    @staticmethod
    def _checkpoint_double(block, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor):
        def fn(img_, txt_, vec_, pe_):
            return block(img=img_, txt=txt_, vec=vec_, pe=pe_)
        return checkpoint(fn, img, txt, vec, pe, use_reentrant=False)

    @staticmethod
    def _checkpoint_single(block, img: Tensor, vec: Tensor, pe: Tensor):
        def fn(img_, vec_, pe_):
            return block(img_, vec=vec_, pe=pe_)
        return checkpoint(fn, img, vec, pe, use_reentrant=False)

    def _mvs_from_double_block(
        self,
        block,
        mvs_block: MultiViewSyncBlock,
        img: Tensor,
        cameras: Tensor,
        num_views: int,
        vec: Tensor,
    ) -> Tensor:
        img_mod1, _ = block.img_mod(vec)
        return mvs_block(
            img,
            cameras=cameras,
            num_views=num_views,
            mod_shift=img_mod1.shift,
            mod_scale=img_mod1.scale,
            mod_gate=img_mod1.gate,
        )

    def _mvs_from_single_block(
        self,
        block,
        mvs_block: MultiViewSyncBlock,
        img_part: Tensor,
        cameras: Tensor,
        num_views: int,
        vec: Tensor,
    ) -> Tensor:
        mod, _ = block.modulation(vec)
        return mvs_block(
            img_part,
            cameras=cameras,
            num_views=num_views,
            mod_shift=mod.shift,
            mod_scale=mod.scale,
            mod_gate=mod.gate,
        )

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        cameras: Tensor | None = None,
        num_views: int = 1,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance-distilled FLUX model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for i, block in enumerate(self.double_blocks):
            if self.training and self._gradient_checkpointing:
                img, txt = self._checkpoint_double(block, img, txt, vec, pe)
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe)
            if cameras is not None and num_views > 1:
                img = self._mvs_from_double_block(
                    block,
                    self.mv_double_blocks[i],
                    img,
                    cameras,
                    num_views,
                    vec,
                )

        txt_len = txt.shape[1]
        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if self.training and self._gradient_checkpointing:
                img = self._checkpoint_single(block, img, vec, pe)
            else:
                img = block(img, vec=vec, pe=pe)

            if self.inject_single_blocks and cameras is not None and num_views > 1 and str(i) in self.mv_single_blocks:
                txt_part, img_part = img[:, :txt_len, :], img[:, txt_len:, :]
                img_part = self._mvs_from_single_block(
                    block,
                    self.mv_single_blocks[str(i)],
                    img_part,
                    cameras,
                    num_views,
                    vec,
                )
                img = torch.cat((txt_part, img_part), dim=1)

        img = img[:, txt_len:, ...]
        img = self.final_layer(img, vec)
        return img


def extract_mv_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k.startswith("mv_double_blocks") or k.startswith("mv_single_blocks")
    }


def load_mv_state_dict(model: nn.Module, path: str, map_location: str | torch.device = "cpu"):
    obj = torch.load(path, map_location=map_location)
    state = obj.get("state_dict", obj)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected
