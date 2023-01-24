import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from .MRIDataset import MRIDataset

def get_loaders(directory: str, train_ratio: float, val_ratio: float, batch_size: int, size=(256, 256), num_workers=2, reload_data = False):
    if reload_data:
        all_samples = get_image_paths(directory, check_mask=True)
        
        train_partition = int(len(all_samples[0]) * train_ratio)
        val_partition = int(len(all_samples[0]) * (train_ratio + val_ratio))

        train = [all_samples[0][:train_partition], all_samples[1][:train_partition]]
        val = [all_samples[0][train_partition:val_partition], all_samples[1][train_partition:val_partition]]
        test = [all_samples[0][val_partition:], all_samples[1][val_partition:]]

        directory = directory.replace('raw', 'processed')
        try:
            shutil.rmtree(os.path.join(directory, 'train'))
        except:
            pass
        try:
            os.mkdir(os.path.join(directory, 'train'))
        except:
            pass
        try:
            shutil.rmtree(os.path.join(directory, 'val'))
        except:
            pass
        try:
            os.mkdir(os.path.join(directory, 'val'))
        except:
            pass
        try:
            shutil.rmtree(os.path.join(directory, 'test'))
        except:
            pass
        try:
            os.mkdir(os.path.join(directory, 'test'))
        except:
            pass

        for i in tqdm(range(len(train[0])), desc='Train'):
            p = os.path.split(train[0][i])
            shutil.copy(train[0][i], os.path.join(os.path.join(os.path.split(p[0])[0].replace('raw', 'processed'), 'train'), p[1]))
            p = os.path.split(train[1][i])
            shutil.copy(train[1][i], os.path.join(os.path.join(os.path.split(p[0])[0].replace('raw', 'processed'), 'train'), p[1]))

        for i in tqdm(range(len(val[0])), desc='Val'):
            p = os.path.split(val[0][i])
            shutil.copy(val[0][i], os.path.join(os.path.join(os.path.split(p[0])[0].replace('raw', 'processed'), 'val'), p[1]))
            p = os.path.split(val[1][i])
            shutil.copy(val[1][i], os.path.join(os.path.join(os.path.split(p[0])[0].replace('raw', 'processed'), 'val'), p[1]))

        for i in tqdm(range(len(test[0])), desc='Test'):
            p = os.path.split(test[0][i])
            shutil.copy(test[0][i], os.path.join(os.path.join(os.path.split(p[0])[0].replace('raw', 'processed'), 'test'), p[1]))
            p = os.path.split(test[1][i])
            shutil.copy(test[1][i], os.path.join(os.path.join(os.path.split(p[0])[0].replace('raw', 'processed'), 'test'), p[1]))

    else:
        train = get_image_paths(directory, 'train', False)
        val = get_image_paths(directory, 'val', False)
        test = get_image_paths(directory, 'test', False)

    train_ds = MRIDataset(train[0], train[1], size)
    val_ds = MRIDataset(val[0], val[1], size)
    test_ds = MRIDataset(test[0], test[1], size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_loader, val_loader, test_loader

def get_image_paths(directory: str, dataset = 'train', check_mask = False) -> 'tuple[list, list]':
    images = []
    masks = []
    assert os.path.isdir(directory), f'{directory} is not a valid directory'
    if not check_mask:
        path = os.path.join(directory, dataset)
        for d in tqdm(sorted(os.listdir(path)), desc=dataset):
            if 'mask' in d:
                masks.append(os.path.join(path, d))
            else:
                images.append(os.path.join(path, d))

        return images, masks

    for d in tqdm(sorted(os.listdir(directory)), desc='Patients'):
        path = os.path.join(directory, d)
        if os.path.isdir(path):
            iters = int(len(os.listdir(path)) / 2)
            for i in range(iters): 
                file = f'{os.path.join(path, d)}_{i + 1}.tif'
                mask = f'{os.path.join(path, d)}_{i + 1}_mask.tif'
                if np.max(cv2.imread(mask)) > 0:
                    images.append(file)
                    masks.append(mask)

    return unison_shuffled_copies(images, masks)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return np.array(a)[p.astype(int)], np.array(b)[p.astype(int)]