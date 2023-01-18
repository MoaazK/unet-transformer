import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union

from utils.Metrics import Metrics
from models import UNet, AttentionUNet, TransformerUNet
from utils.config import ACCURACY, AUPRC, AUROC_, DICE_SCORE, F1_SCORE, JACCARD_INDEX, PRECISION, RECALL, SPECIFICITY

@torch.inference_mode()
def evaluate(
    model: Union[UNet, AttentionUNet, TransformerUNet],
    criterion: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    metrics: Metrics):

    total = 0
    all_loss = 0
    avg_sum = 0
    model.to(device=device)
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validation')
    for batch_idx, (X, Y) in pbar:
        X, Y = Variable(X.to(device=device)), Y.to(device=device)
        batch_size = X.size(0)

        logits: torch.Tensor = model(X)
        loss = criterion(logits, Y)

        loss = loss.detach()
        l = loss.item()
        avg_sum += l
        all_loss += l * batch_size
        total += batch_size

        metric_dict = metrics.update(logits, Y)
        pbar.set_postfix(
            Loss=f'{avg_sum / (batch_idx + 1):.4f}',
            Accuracy=f'{metric_dict[ACCURACY]:.4f}',
            Dice=f'{metric_dict[DICE_SCORE]:.4f}',
            IoU=f'{metric_dict[JACCARD_INDEX]:.4f}'
        )
    
    return all_loss / total
