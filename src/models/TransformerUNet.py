from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .EncoderDecoder import *
from src.utils.config import *

DEVICE = get_device()

class TransformerUNet(nn.Module):
    def __init__(self, input_dim: int, channels: Tuple[int], is_residual: bool = False, num_heads = 2, bias=False) -> None:
        super(TransformerUNet, self).__init__()

        self.channels = channels
        self.pos_encoding = PositionalEncoding()
        self.encode = nn.ModuleList([EncoderLayer(channels[i], channels[i + 1], is_residual) for i in range(len(channels) - 2)])
        self.bottle_neck = ConvBlock(channels[-2], channels[-1], is_residual)
        self.mhsa = MultiHeadSelfAttention((input_dim // 2**(len(channels)-2))**2, num_heads, bias)
        self.mhca = nn.ModuleList([MultiHeadCrossAttention((input_dim // 2**(i-1))**2, num_heads, channels[i + 1], channels[i], bias) for i in reversed(range(1, len(channels) - 1))])
        self.decode = nn.ModuleList([DecoderLayer(channels[i + 1], channels[i], is_residual) for i in reversed(range(1, len(channels) - 1))])
        self.output = nn.Conv2d(channels[1], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_x_list: List[torch.Tensor] = []
        for i in range(len(self.channels) - 2):
            skip_x, x = self.encode[i](x)
            skip_x_list.append(skip_x)

        x = self.bottle_neck(x)
        x = self.pos_encoding(x)
        x = self.mhsa(x)

        for i, skip_x in enumerate(reversed(skip_x_list)):
            x = self.pos_encoding(x)
            skip_x = self.pos_encoding(skip_x)
            skip_x = self.mhca[i](x, skip_x)
            x = self.decode[i](x, skip_x)

        return self.output(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias=False) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        # self.pos_enc = PositionalEncoding()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # x = self.pos_enc(x)
        x = x.permute(0, 2, 3, 1).view((b, h * w, c))
        x, _ = self.mha(x, x, x, need_weights=False)
        return x.view((b, h, w, c)).permute(0, 3, 1, 2)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, channel_S: int, channel_Y: int, bias=False) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        # self.pos_enc_S = PositionalEncoding()
        # self.pos_enc_Y = PositionalEncoding()

        self.conv_S = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channel_S, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU(inplace=True)
        )

        self.conv_Y = nn.Sequential(
            nn.Conv2d(channel_Y, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU(inplace=True)
        )

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

        self.upsample = nn.Sequential(
            nn.Conv2d(channel_S, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.Sigmoid(),
            nn.ConvTranspose2d(channel_S, channel_S, 2, 2, bias=bias)
        )

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # s_enc = self.pos_enc_S(s)
        # y_enc = self.pos_enc_Y(y)
        
        # s = self.conv_S(s_enc)
        # y = self.conv_Y(y_enc)

        s_enc = s
        s = self.conv_S(s)
        y = self.conv_Y(y)

        b, c, h, w = s.size()
        s = s.permute(0, 2, 3, 1).view((b, h * w, c))

        b, c, h, w = y.size()
        y = y.permute(0, 2, 3, 1).view((b, h * w, c))

        y, _ = self.mha(y, y, s, need_weights=False)
        y = y.view((b, h, w, c)).permute(0, 3, 1, 2)
        
        y = self.upsample(y)

        return torch.mul(y, s_enc)

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super(PositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos_encoding = self.positional_encoding(h * w, c)
        x = x.view((b, c, h * w)) + pos_encoding
        return x.view((b, c, h, w))

    def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
        depth = depth / 2

        positions = torch.arange(length, dtype=DTYPE, device=DEVICE)
        depths = torch.arange(depth, dtype=DTYPE, device=DEVICE) / depth

        angle_rates = 1 / (10000**depths)
        angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

        pos_encoding = torch.cat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)

        return pos_encoding

# class PositionalEncoding(nn.Module):
#     def __init__(self, channels_in: int) -> None:
#         super(PositionalEncoding, self).__init__()

#         channels_in = int(torch.ceil(channels_in / 2))
#         self.channels_in = channels_in
#         angle_rates = 1. / (10000**(torch.arange(0, channels_in, 2, dtype=DTYPE) / channels_in))
#         self.register_buffer('angle_rates', angle_rates)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, h, w = x.size()
#         pos_x = torch.arange(h, dtype=DTYPE, device=DEVICE)
#         pos_y = torch.arange(w, dtype=DTYPE, device=DEVICE)
#         angle_rads_x = torch.einsum('i,j->ij', pos_x, self.angle_rates)
#         angle_rads_y = torch.einsum('i,j->ij', pos_y, self.angle_rates)
#         pos_enc_x = torch.cat((torch.sin(angle_rads_x), torch.cos(angle_rads_x)), dim=-1).unsqueeze(1)
#         pos_enc_y = torch.cat((torch.sin(angle_rads_y), torch.cos(angle_rads_y)), dim=-1)
#         pos_enc = torch.zeros((h, w, self.channels_in * 2), dtype=DTYPE, device=DEVICE)
#         pos_enc[:, :, :self.channels_in] = pos_enc_x
#         pos_enc[:, :, self.channels_in:self.channels_in * 2] = pos_enc_y
#         pos_enc = pos_enc[None, :, :, c].repeat(b, 1, 1, 1).permute(0, 3, 1, 2)

#         return x + pos_enc

#     def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
#         depth = depth / 2

#         positions = torch.arange(length, dtype=DTYPE, device=DEVICE)
#         depths = torch.arange(depth, dtype=DTYPE, device=DEVICE) / depth

#         angle_rates = 1 / (10000**depths)
#         angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

#         pos_encoding = torch.cat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)

#         return pos_encoding
