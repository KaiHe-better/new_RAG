import torch

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA is available. GPU support is enabled.")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. GPU support is not enabled.")