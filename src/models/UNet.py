from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .EncoderDecoder import *

class UNet(nn.Module):
    def __init__(self, channels: Tuple[int], is_residual: bool = False) -> None:
        super(UNet, self).__init__()

        self.channels = channels
        self.encode = nn.ModuleList([EncoderLayer(channels[i], channels[i + 1], is_residual) for i in range(len(channels) - 2)])
        self.bottle_neck = ConvBlock(channels[-2], channels[-1], is_residual)
        self.decode = nn.ModuleList([DecoderLayer(channels[i + 1], channels[i], is_residual) for i in reversed(range(1, len(channels) - 1))])
        self.output = nn.Conv2d(channels[1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_x_list: List[torch.Tensor] = []
        for i in range(len(self.channels) - 2):
            skip_x, x = self.encode[i](x)
            skip_x_list.append(skip_x)

        x = self.bottle_neck(x)

        for i, skip_x in enumerate(reversed(skip_x_list)):
            x = self.decode[i](x, skip_x)

        return self.output(x)
