#!/usr/bin/env python

#################################
#                               #
#        AI4Mars Dataset        #
#    created by Warren Kozak    #
#                               #
#################################

import cv2
import os
import torch

import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2

class AI4MarsDataset(Dataset):
    '''A class for the custom AI4Mars Dataset'''

    def __init__(self, img_dir, labels_dir, train_transform=None, val_transform=None):

        self.img_dir = img_dir
        self.labels_dir = labels_dir

        self.img_path = Path(img_dir)
        self.labels_path = Path(labels_dir)

        self.train_transform = train_transform
        self.val_transform = val_transform

        self.validation_enabled = False

        # Create an array containing all images and their paired label
        self.pairs = []
        for item in sorted(self.img_path.iterdir()):
            label = self.labels_path / item.with_suffix(".png").name
            if label.is_file():
                self.pairs.append((item, label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        img = cv2.imread(self.pairs[idx][0], cv2.IMREAD_COLOR)
        label = cv2.imread(self.pairs[idx][1], cv2.IMREAD_GRAYSCALE)

        name = self.pairs[idx][0].stem

        label = np.where(label == 255, 0, label + 1).astype(np.uint8)

        if self.train_transform and not self.validation_enabled:
            out = self.train_transform(image=img, mask=label)
            img, label = out['image'], out['mask']

        if self.val_transform and self.validation_enabled:
            out = self.val_transform(image=img, mask=label)
            img, label = out['image'], out['mask']

        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.from_numpy(label).long()

        return img, label, name