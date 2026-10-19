"""
File created by: Noah Lee
This handles what happens with image data when being fetched by the dataloader in train.py.
Add image augmentation in __get_item__()
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# ImageNet-1k mean, std stats - pretrained encoder expects these
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# TODO: add data augmentation, if dataset is a training dataset, we want to augment the image when we get it from __get_item__

class MarsDataset(Dataset):
    def __init__(self, root, size=576):
        root = Path(root)
        self.size = size
        self.images = sorted((root / "images").glob("*.JPG"))
        # pair by filename
        self.labels = [next((root / "labels_remapped").glob(f"{p.stem}_*.png")) for p in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mask = Image.open(self.labels[idx]).resize((self.size, self.size), Image.NEAREST)  # NEAREST keeps labels as ints
        img = (np.asarray(img, dtype=np.float32) / 255. - MEAN) / STD
        mask = np.asarray(mask, dtype=np.int64)  # 0-3 classes, 255 unlabeled
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask)