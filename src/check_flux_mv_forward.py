from __future__ import annotations

import argparse
import torch

from src.models.flux_multiview_loader import load_multiview_flux
from src.training.flux_train_utils import make_img_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="flux-dev")
    p.add_argument("--num_views", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--txt_len", type=int, default=16)
    p.add_argument("--mv_adapter_dim", type=int, default=128)
    p.add_argument("--hf_download", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    model = load_multiview_flux(
        name=args.model_name,
        device=device,
        dtype=dtype,
        hf_download=args.hf_download,
        mv_adapter_dim=args.mv_adapter_dim,
    )
    model.freeze_base_model()
    model.eval()

    b, v, s = 1, args.num_views, args.seq_len
    img = torch.randn(b * v, s, 64, device=device, dtype=dtype)
    # fake IDs on a square grid if possible
    side = int(s ** 0.5)
    if side * side == s:
        img_ids = make_img_ids(b * v, side * 2, side * 2, device, dtype)
    else:
        img_ids = torch.zeros(b * v, s, 3, device=device, dtype=dtype)
    txt = torch.randn(b * v, args.txt_len, 4096, device=device, dtype=dtype)
    txt_ids = torch.zeros(b * v, args.txt_len, 3, device=device, dtype=dtype)
    y = torch.randn(b * v, 768, device=device, dtype=dtype)
    timesteps = torch.ones(b * v, device=device, dtype=dtype)
    guidance = torch.full((b * v,), 3.5, device=device, dtype=dtype)
    cameras = torch.zeros(b, v, 12, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(img, img_ids, txt, txt_ids, timesteps, y, guidance, cameras=cameras, num_views=v)
    print("forward ok", tuple(out.shape))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params", trainable)


if __name__ == "__main__":
    main()
