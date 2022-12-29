import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
import copy

import matplotlib
import matplotlib.pyplot as plt
import torchmetrics
matplotlib.use('Agg')

from data.make_dataset import ImageDataset
from models import UNet, AttentionUNet

def train_model(model, train_loader, val_loader, criterion, epochs, optimizer, device):
    best_model = copy.deepcopy(model)
    train_losses = []
    val_losses = []
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        dice = torchmetrics.Dice().to(device)
        for x,y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            optimizer.zero_grad()
            x, y = x.to(device).float(), y.to(device).float()
            logits = model(x)
            loss = criterion(logits, y) + 1 - dice(logits, y.int())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        no_corrects = 0
        total = 0
        val_loss = 0
        total_pixels = 0
        metric = torchmetrics.Accuracy().to(device)
        dice = torchmetrics.Dice().to(device)
        for x,y in val_loader:
            with torch.no_grad():
                x, y = x.to(device).float(), y.to(device).float()
                logits = model(x)
                loss = criterion(logits, y) + 1 - dice(logits, y.int())
                y = y.int()
                val_loss += loss.item() * x.shape[0]
                preds = torch.round(torch.sigmoid(logits))
                no_corrects += torch.sum(preds == y).item()
                total_pixels += torch.prod(torch.tensor(y.shape)).item()
                total += x.shape[0]
                metric(preds, y)
                dice(preds, y)
                
        val_acc = no_corrects / total_pixels
        print(f"Epoch {epoch+1}, validation accuracy: {val_acc}, my acc: {metric.compute()}, dice: {dice.compute()}")
        val_losses.append(val_loss / total)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
    
    return best_model, train_losses, val_losses

def plot_losses(train_losses, val_losses):
    plt.figure()
    n_epochs = len(val_losses)
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, val_losses, label='val loss')
    plt.legend()
    plt.title('Training Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.tight_layout()
    plt.savefig('segmentation_plot.png')

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')
    model = AttentionUNet((1, 32, 64, 128, 256, 512, 1024), True).to(device)
    # print(model)
    # summary(model, (1, 256, 256), 4)
    train_dataset = ImageDataset("data/raw/Br35H-Mask-RCNN/TRAIN", "data/raw/Br35H-Mask-RCNN/TRAIN_MASK")
    val_dataset = ImageDataset("data/raw/Br35H-Mask-RCNN/VAL", "data/raw/Br35H-Mask-RCNN/VAL_MASK")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1
    best_model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, epochs, optimizer, device)
    # plot_losses(train_losses, val_losses)
    # torch.save(best_model, "best_model.pt")