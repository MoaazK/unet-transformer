import numpy as np
import os
from tqdm import tqdm
import cv2
from torch.utils.data import DataLoader

from .MRIDataset import MRIDataset

def get_loaders(dir: str, train_ratio: float, val_ratio: float, batch_size: int):
    all_samples = get_image_paths(dir)
    
    train_partition = int(len(all_samples[0]) * train_ratio)
    val_partition = int(len(all_samples[0]) * (train_ratio + val_ratio))

    train = [all_samples[0][:train_partition], all_samples[1][:train_partition]]
    val = [all_samples[0][train_partition:val_partition], all_samples[1][train_partition:val_partition]]
    test = [all_samples[0][val_partition:], all_samples[1][val_partition:]]

    size = (256, 256)

    train_ds = MRIDataset(train[0], train[1], size)
    val_ds = MRIDataset(val[0], val[1], size)
    test_ds = MRIDataset(test[0], test[1], size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=True)

    return train_loader, val_loader, test_loader

def get_image_paths(dir: str) -> 'tuple[list, list]':
    images = []
    masks = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    for d in tqdm(sorted(os.listdir(dir)), desc='Patients'):
        path = os.path.join(dir, d)
        if os.path.isdir(path):
            iters = int(len(os.listdir(path)) / 2)
            for i in range(iters): 
                file = f'{os.path.join(path, d)}_{i + 1}.tif'
                mask = f'{os.path.join(path, d)}_{i + 1}_mask.tif'
                if np.max(cv2.imread(mask)) > 0:
                    images.append(file)
                    masks.append(mask)

    return images, masks
