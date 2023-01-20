from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.EncoderDecoder import ConvBlock, EncoderLayer, DecoderLayer

class AttentionUNet(nn.Module):
    def __init__(self, channels: Tuple[int], is_residual: bool = False, bias = False) -> None:
        super(AttentionUNet, self).__init__()

        self.channels = channels
        self.encode = nn.ModuleList([EncoderLayer(channels[i], channels[i + 1], is_residual, bias) for i in range(len(channels) - 2)])
        self.bottle_neck = ConvBlock(channels[-2], channels[-1], is_residual, bias)
        self.attention_gate = nn.ModuleList([AttentionGate(channels[i + 1], channels[i], channels[i], bias) for i in reversed(range(1, len(channels) - 1))])
        self.decode = nn.ModuleList([DecoderLayer(channels[i + 1], channels[i], is_residual, bias) for i in reversed(range(1, len(channels) - 1))])
        self.output = nn.Conv2d(channels[1], 1, 1)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_x_list: List[torch.Tensor] = []
        for i in range(len(self.channels) - 2):
            skip_x, x = self.encode[i](x)
            skip_x_list.append(skip_x)

        x = self.bottle_neck(x)

        for i, skip_x in enumerate(reversed(skip_x_list)):
            skip_x = self.attention_gate[i](skip_x, x)
            x = self.decode[i](skip_x, x)

        return self.output(x)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int, bias=False) -> None:
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=bias),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, 2, bias=bias),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=bias).apply(lambda m: nn.init.xavier_uniform_(m.weight.data)),
            nn.BatchNorm2d(1)
        )

        # self.upsample = nn.Upsample(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(1, 1, 2, 2, bias=bias)

        self.final_conv = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=bias),
            nn.BatchNorm2d(F_int),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        phi_g = self.W_g(g)
        theta_x = self.W_x(x)

        theta_x = torch.add(phi_g, theta_x)
        theta_x = F.relu(theta_x)
        theta_x = self.psi(theta_x)
        theta_x = torch.sigmoid(theta_x)
        theta_x = self.upsample(theta_x)
        x = torch.mul(x, theta_x)

        return self.final_conv(x)
