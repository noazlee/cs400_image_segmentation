import cv2
import os
import torch

from torch.utils.data import Dataset
from torchvision.transforms import v2

class AI4MarsDataset(Dataset):
    '''A class for the custom AI4Mars Dataset'''

    def __init__(self, img_dir, labels_dir, transform=None):

        self.img_dir = img_dir
        self.labels_dir = labels_dir

        self.image_filenames = sorted(os.listdir(img_dir))
        self.label_filenames = sorted(os.listdir(labels_dir))

        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        label_path = os.path.join(self.labels_dir, self.label_filenames[idx])

        img = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            out = self.transform(image=img, mask=label)
            img, label = out['image'], out['mask']

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = v2.functional.to_dtype(img, dtype=torch.float32, scale=True)
        label = torch.from_numpy(label).long()

        return img, label