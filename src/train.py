import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union

from utils.Metrics import Metrics
from models import UNet, AttentionUNet, TransformerUNet
from utils.config import ACCURACY, AUPRC, AUROC_, DICE_SCORE, F1_SCORE, JACCARD_INDEX, PRECISION, RECALL, SPECIFICITY

def train(
    model: Union[UNet, AttentionUNet, TransformerUNet],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dataloader: DataLoader,
    metrics: Metrics):

    loss_sum = 0
    loss_total = 0
    avg_sum = 0
    train_loss = []
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training  ')
    for batch_idx, (X, Y) in pbar:
        optimizer.zero_grad()
        X, Y = Variable(X.to(device=device)), Y.to(device=device)
        batch_size = X.size(0)

        logits: torch.Tensor = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        loss = loss.detach()
        l = loss.item()
        avg_sum += l
        loss_sum += l * batch_size
        loss_total += batch_size
        train_loss.append(avg_sum / (batch_idx + 1))

        metric_dict = metrics.update(logits, Y)
        pbar.set_postfix(
            Loss=f'{train_loss[-1]:.4f}',
            Accuracy=f'{metric_dict[ACCURACY]:.4f}',
            Dice=f'{metric_dict[DICE_SCORE]:.4f}',
            IoU=f'{metric_dict[JACCARD_INDEX]:.4f}'
        )

    return train_loss, loss_sum / loss_total
