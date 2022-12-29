from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .EncoderDecoder import *

class TransformerUNet(nn.Module):
    def __init__(self, channels: Tuple[int], is_residual: bool = False) -> None:
        super(TransformerUNet, self).__init__()