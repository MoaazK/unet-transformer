import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, source_paths: list, target_paths: list, size = (256, 256)) -> None:
        self.source_paths = source_paths
        self.target_paths = target_paths

        self.source_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size)
        ])
        self.target_transforms = transforms.Compose([
            transforms.ToTensor(),
            BinaryTransform(0.5),
            transforms.Resize(size, transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self) -> int:
        return len(self.source_paths)

    def __getitem__(self, index: int) -> 'tuple[transforms.Compose, transforms.Compose]':
        image_path = self.source_paths[index]
        mask_path = self.target_paths[index]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        image = self.source_transforms(image)
        mask = self.target_transforms(mask)

        return image, mask

class BinaryTransform(object):
    def __init__(self, threshold) -> None:
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x > self.threshold).to(x.dtype)
