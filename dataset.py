import os
import tifffile

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define Dataset
class CellDataset(Dataset):
    def __init__(self, data_path, transform=None, transform_mask=None):
        self.transform = transform
        self.transform_mask = transform_mask
        self.data_path = data_path
        self.data = os.listdir(data_path)
        self.data = [i for i in self.data if not i.endswith('_mask.tif')]
        self.data.sort()
        self.data = [os.path.join(data_path, i) for i in self.data]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = tifffile.imread(self.data[idx])
        img = img.astype(np.float32)
        # print(img[0][0])
        img = np.transpose(img, (1, 2, 0))
        mask = tifffile.imread(self.data[idx][:-4] + '_mask.tif')
        img[0] = np.clip(img[0], 300, 28000)
        img[1] = np.clip(img[1], 180, 10800)
        img[2] = np.clip(img[2], 1800, 14400)
        img[3] = np.clip(img[3], 1800, 11800)
        # print(mask)
        if self.transform:
            img = self.transform(img)
            mask = self.transform_mask(mask)
        return img, mask
    

# Define test Dataset
class CellDataset_test(Dataset):
    def __init__(self, data_path, transform=None, transform_mask=None):
        self.transform = transform
        self.transform_mask = transform_mask
        self.data_path = data_path
        self.data = os.listdir(data_path)
        self.data = [i for i in self.data if not i.endswith('_mask.tif')]
        self.data.sort()
        self.data = [os.path.join(data_path, i) for i in self.data]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = tifffile.imread(self.data[idx])
        img = img.astype(np.float32)
        # print(img[0][0])
        img = np.transpose(img, (1, 2, 0))
        mask = tifffile.imread(self.data[idx][:-4] + '_mask.tif')
        img[0] = np.clip(img[0], 300, 28000)
        img[1] = img[0]
        img[2] = img[0]
        img[3] = img[0]
        # print(mask)
        if self.transform:
            img = self.transform(img)
            mask = self.transform_mask(mask)
        return img, mask
    
