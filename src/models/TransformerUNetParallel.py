from typing import List, Tuple
import torch
import torch.nn as nn

from utils import config
from models.EncoderDecoder import ConvBlock, EncoderLayer, DecoderLayer

DEVICE = config.get_device()

class TransformerUNetParallel(nn.Module):
    def __init__(self, channels: Tuple[int], num_heads: int = 2, image_size: int = 128, is_residual: bool = False, bias = False) -> None:
        super(TransformerUNetParallel, self).__init__()

        self.channels = channels
        self.cuda_cores = [f'cuda:{i+1}' for i in reversed(range(1, len(channels) - 1))]
        # self.pos_encoding = PositionalEncoding().to('cuda:0')
        self.encode = nn.ModuleList([EncoderLayer(channels[i], channels[i + 1], is_residual, bias).to('cuda:6') for i in range(len(channels) - 2)])
        self.bottle_neck = ConvBlock(channels[-2], channels[-1], is_residual, bias).to('cuda:6')
        self.pos_embed_mhsa = nn.Parameter(torch.randn(1, channels[-1], image_size // (2**(len(channels) - 2)), image_size // (2**(len(channels) - 2)))).to('cuda:6')
        self.mhsa = MultiHeadSelfAttention(channels[-1], num_heads, bias).to('cuda:1')
        self.pos_embed_x = nn.ParameterList([nn.Parameter(torch.randn(1, channels[i + 1], image_size // (2**i), image_size // (2**i))).to('cuda:0') for i in reversed(range(1, len(channels) - 1))])
        self.pos_embed_skip_x = nn.ParameterList([nn.Parameter(torch.randn(1, channels[i], image_size // (2**(i - 1)), image_size // (2**(i - 1)))).to('cuda:0') for i in reversed(range(1, len(channels) - 1))])
        self.mhca = nn.ModuleList([MultiHeadCrossAttention(channels[i], num_heads, channels[i], channels[i + 1], bias).to(f'cuda:{i+1}') for i in reversed(range(1, len(channels) - 1))])
        self.decode = nn.ModuleList([DecoderLayer(channels[i + 1], channels[i], is_residual, bias).to('cuda:7') for i in reversed(range(1, len(channels) - 1))])
        self.output = nn.Conv2d(channels[1], 1, 1).to('cuda:0')

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_x_list: List[torch.Tensor] = []
        for i in range(len(self.channels) - 2):
            skip_x, x = self.encode[i](x.to('cuda:6'))
            skip_x_list.append(skip_x)

        x = self.bottle_neck(x.to('cuda:6'))
        # x = self.pos_encoding(x.to('cuda:0'))
        x = x.to('cuda:6') + self.pos_embed_mhsa
        x = self.mhsa(x.to('cuda:1'))

        for i, skip_x in enumerate(reversed(skip_x_list)):
            # x = self.pos_encoding(x.to('cuda:0'))
            # skip_x = self.pos_encoding(skip_x.to('cuda:0'))
            x = x.to('cuda:0') + self.pos_embed_x[i]
            skip_x = skip_x.to('cuda:0') + self.pos_embed_skip_x[i]
            skip_x = self.mhca[i](skip_x.to(self.cuda_cores[i]), x.to(self.cuda_cores[i]))
            x = self.decode[i](skip_x.to('cuda:7'), x.to('cuda:7'))

        return self.output(x.to('cuda:0'))
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias=False) -> None:
        super(MultiHeadSelfAttention, self).__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).reshape((b, h * w, c))
        x, _ = self.mha(x, x, x, need_weights=False)
        return x.reshape((b, h, w, c)).permute(0, 3, 1, 2)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, channel_S: int, channel_Y: int, bias=False) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        self.conv_S = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(channel_S, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        )

        self.conv_Y = nn.Sequential(
            nn.Conv2d(channel_Y, channel_S, 1, bias=bias),
            nn.BatchNorm2d(channel_S),
            nn.ReLU()
        )

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

        self.upsample = nn.Sequential(
            nn.Conv2d(channel_S, channel_S, 1, bias=bias).apply(lambda m: nn.init.xavier_uniform_(m.weight.data)),
            nn.BatchNorm2d(channel_S),
            nn.Sigmoid(),
            nn.ConvTranspose2d(channel_S, channel_S, 2, 2, bias=bias)
        )

    def forward(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s_enc = s
        s = self.conv_S(s)
        y = self.conv_Y(y)

        b, c, h, w = s.size()
        s = s.permute(0, 2, 3, 1).reshape((b, h * w, c))

        b, c, h, w = y.size()
        y = y.permute(0, 2, 3, 1).reshape((b, h * w, c))

        y, _ = self.mha(y, y, s, need_weights=False)
        y = y.reshape((b, h, w, c)).permute(0, 3, 1, 2)
        
        y = self.upsample(y)

        return torch.mul(y, s_enc)

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super(PositionalEncoding, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        pos_encoding = self.positional_encoding(h * w, c)
        pos_encoding = pos_encoding.permute(1, 0).unsqueeze(0).repeat(b, 1, 1)
        x = x.reshape((b, c, h * w)) + pos_encoding
        return x.reshape((b, c, h, w))

    def positional_encoding(self, length: int, depth: int) -> torch.Tensor:
        depth = depth / 2

        positions = torch.arange(length, dtype=config.DTYPE, device=DEVICE)
        depths = torch.arange(depth, dtype=config.DTYPE, device=DEVICE) / depth

        angle_rates = 1 / (10000**depths)
        angle_rads = torch.einsum('i,j->ij', positions, angle_rates)

        pos_encoding = torch.cat((torch.sin(angle_rads), torch.cos(angle_rads)), dim=-1)

        return pos_encoding
