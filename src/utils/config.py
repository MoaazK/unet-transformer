import torch

DTYPE = torch.float32
USE_GPU = True

def get_device() -> torch.device:
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    return device
