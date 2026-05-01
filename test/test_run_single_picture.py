import torch
from diffusers import FluxPipeline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
prompt = "in the playground, a group of children are playing with a ball, the sun is shining brightly, and the sky is clear blue. The children are laughing and having fun, enjoying their time together in the warm weather. The playground is filled with colorful equipment, such as swings, slides, and climbing structures, creating a lively and joyful atmosphere."
model_path = "/data/jwren/diffusion_models/FLUX.1-dev/"
# model_path = '/data/model_cjh/FLUX.2-dev/' #unuseable for now

if torch.cuda.is_available():
    #TODO: change this to the GPU you want to use
    device = "cuda:0"
    print("CUDA is available. Running on GPU.")
else:
    raise ValueError("CUDA is not available. Please check your PyTorch installation and GPU setup.")
    
pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=40,
    max_sequence_length=512,
    generator=torch.Generator(device=device).manual_seed(0) # for reproducibility
).images[0]
save_path = "./test/album/flux-dev41.png"
image.save(save_path)