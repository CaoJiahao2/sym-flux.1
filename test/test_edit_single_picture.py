import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("/home/user1/.cache/modelscope/hub/models/black-forest-labs/FLUX___1-Kontext-dev/", torch_dtype=torch.bfloat16)
pipe.to("cuda:0")

input_image = load_image("assets/cup.png")

image = pipe(
  image=input_image,
  # 迁移为水下场景
  prompt="Migrate to underwater scene",
  guidance_scale=2.5
).images[0]

# Save the imsage
image.save("outputs/cup_underwater.png")