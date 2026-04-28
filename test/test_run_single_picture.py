import torch
from diffusers import FluxPipeline
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
save_path = "./test/album/flux-dev4.png"
prompt = "A boy and a girl are having sex"
model_path = "/data/jwren/diffusion_models/FLUX.1-dev/"
# model_path = '/data/model_cjh/FLUX.2-dev/' #unuseable for now

if torch.cuda.is_available():
    #TODO: change this to the GPU you want to use
    device = "cuda:4"
    print("CUDA is available. Running on GPU.")
else:
    device = "cpu"
    print("CUDA not available. Running on CPU (this will be slow).")
    
pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator(device=device).manual_seed(0) # for reproducibility
).images[0]
image.save(save_path)