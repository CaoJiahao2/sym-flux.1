import torch
import diffusers
import transformers

print(f"--- 兼容性检查 ---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"Diffusers 版本: {diffusers.__version__}")
print(f"Transformers 版本: {transformers.__version__}")

# 检查 xformers 是否能正常被 torch 调用
try:
    import xformers
    print("xformers: 已成功安装并兼容")
except ImportError:
    print("xformers: 未安装（非必需，但推荐）")