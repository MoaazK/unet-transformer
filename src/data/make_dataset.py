import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

class ImageDataset(Dataset):
    
    def __init__(self, source_root, target_root):
        self.source_paths = sorted(make_dataset(source_root))
        self.target_paths = sorted(make_dataset(target_root))
        self.source_transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((128,128))]) # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.target_transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((128,128), InterpolationMode.NEAREST)]) # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def __len__(self):
        return len(self.source_paths)
    
    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('L')
        
        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('L')
        
        from_im = self.source_transforms(from_im)
        to_im = self.target_transforms(to_im)
        return from_im, to_im

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images