import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_residual: bool = False, bias = False) -> None:
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

        if is_residual:
            self.conv_skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=bias),
                nn.BatchNorm2d(out_channels)
            )

        self.is_residual = is_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shortcut = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.is_residual:
            x_shortcut = self.conv_skip(x_shortcut)
            x = torch.add(x, x_shortcut)

        x = F.relu(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_residual: bool = False, bias=False) -> None:
        super(EncoderLayer, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels, is_residual, bias)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        pool = self.pool(x)
        return x, pool

class DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_residual: bool = False, bias=False) -> None:
        super(DecoderLayer, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, bias=bias)
        self.conv = ConvBlock(in_channels, out_channels, is_residual, bias)

    def forward(self, skip_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = self.transpose(x)
        x = torch.cat((skip_x, x), dim=1)
        return self.conv(x)
