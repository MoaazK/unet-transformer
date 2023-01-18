import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Dice

class DiceBCELoss(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super(DiceBCELoss, self).__init__()

        self.device = device

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dice = Dice().to(device=self.device)
        return F.binary_cross_entropy_with_logits(input, target) + 1 - dice(torch.round(torch.sigmoid(input)), target.int())
