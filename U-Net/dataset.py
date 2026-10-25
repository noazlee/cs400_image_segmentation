import cv2
import os
import torch

from pathlib import Path

from torch.utils.data import Dataset
from torchvision.transforms import v2

class AI4MarsDataset(Dataset):
    '''A class for the custom AI4Mars Dataset'''

    def __init__(self, img_dir, labels_dir, transform=None):

        self.img_dir = img_dir
        self.labels_dir = labels_dir

        self.img_path = Path(img_dir)
        self.labels_path = Path(labels_dir)

        self.transform = transform

        # Create an array containing all images and their paired label
        self.pairs = []
        for item in self.img_path.iterdir():
            label = self.labels_path / item.with_suffix(".png").name
            if label.is_file():
                self.pairs.append((item, label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        img = cv2.imread(self.pairs[idx][0], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.pairs[idx][1], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            out = self.transform(image=img, mask=label)
            img, label = out['image'], out['mask']

        img = torch.from_numpy(img).unsqueeze(0)
        img = v2.functional.to_dtype(img, dtype=torch.float32, scale=True)
        label = torch.from_numpy(label).long()

        # Shift class labels up one
        label = torch.where(label == 255, label, label + 1)

        # Map 255 to 0 class label
        label = torch.where(label == 255, 0, label)

        return img, label