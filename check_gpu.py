import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0))
print("Allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
print("Reserved:", torch.cuda.memory_reserved()/1024**2, "MB")