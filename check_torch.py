import torch

print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"CUDA available: {torch.cuda.is_available()}")
