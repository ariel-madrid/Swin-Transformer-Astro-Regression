import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset
import torch
import numpy as np
import os

class NpyDataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        self.npy_dir = npy_dir
        self.transform = transform
        self.npy_files = sorted([os.path.join(npy_dir, f) for f in os.listdir(npy_dir) if f.endswith(".npy")])

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        data = np.load(self.npy_files[idx], allow_pickle=True).item()
        image = torch.tensor(data["image"], dtype=torch.float32).squeeze()  # (2, H, W)
        params = torch.tensor(data["params"], dtype=torch.float32)  # (6,)

        if self.transform:
            image = self.transform(image)

        return image, params

