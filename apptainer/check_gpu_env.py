# GPU environment check
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Num GPUs: {torch.cuda.device_count()}")

# CUDA and cuDNN versions
print(f"CUDA version (PyTorch built with): {torch.version.cuda}")
if torch.backends.cudnn.is_available():
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
else:
    print("cuDNN version: N/A (not available)")

# NCCL version (used for multi-GPU communication)
if torch.distributed.is_nccl_available():
    print(f"NCCL version: {torch.cuda.nccl.version()}")
else:
    print("NCCL version: N/A (not available)")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU[{i}]: {torch.cuda.get_device_name(i)}")
