import torch
# 1. Check if CUDA is available
print(torch.cuda.is_available())  # Should output True; otherwise, the CUDA environment is completely unavailable
# 2. Check if the GPU is detected
print(torch.cuda.device_count())  # Should be ≥ 1; otherwise, the GPU is not recognized by the driver
print(torch.cuda.get_device_name(0))  # Should output the GPU model (e.g., Tesla V100)