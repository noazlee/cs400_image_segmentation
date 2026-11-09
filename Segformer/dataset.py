"""
File created by: Noah Lee

Modified by Isaac Lund to add data augmentation to the training data as in the SegFormer paper.
Modified by Isaac Lund to add split each image into four non-overlapping 512x512 quadrants for validation.

This handles what happens with image data when being fetched by the dataloader in train.py.
"""

from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "data_preprocessing"))
from dustification import get_params, apply_martian_dust_storm

# ImageNet-1k mean, std stats - pretrained encoder expects these
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Fill padded RGB pixels with approximately the ImageNet mean, used in _pad_to_crop_size() below
PAD_RGB = tuple(np.round(MEAN * 255).astype(np.uint8).tolist())

class MarsDataset(Dataset):
    def __init__(self, root, size=512, training=False, distortion=False, unknown_class=4):
        root = Path(root)
        self.size = size
        self.images = sorted((root / "images").glob("*.JPG"))
        # pair by filename: images/<name>.JPG <-> labels/<name>.png
        self.labels = [root / "labels" / f"{p.stem}.png" for p in self.images]
        self.training = training
        self.distortion = distortion
        # unlabelled 255 pixels are trained as this "none" class. Done before padding,
        # so padded pixels stay 255 and are still ignored by the loss.
        self.unknown_class = unknown_class

    def __len__(self):
        return len(self.images)

    #This functions handles the case where we resize the image to be smaller than 512x512
    #and we need to pad it to ensure a random 512x512 crop is possible---the crop is no longer 
    #random in this case.
    def _pad_to_crop_size(self, img, mask):
        width, height = img.size
        pad_right = max(0, self.size - width)
        pad_bottom = max(0, self.size - height)
        
        #image doesn't need any padding to be cropped to 512x512
        if pad_right == 0 and pad_bottom == 0:
            return img, mask
        
        padded_img = Image.new("RGB",(width + pad_right, height + pad_bottom),color=PAD_RGB)
        #Paste the original image onto the padded image at the top-left corner.
        padded_img.paste(img, (0, 0))

        padded_mask = Image.new(mask.mode, (width + pad_right, height + pad_bottom), color=255)
        #Paste the original mask onto the padded mask at the top-left corner.
        padded_mask.paste(mask, (0, 0))

        return padded_img, padded_mask

    def _dust(self, img):
        # 25% clean, 25% light, 25% medium, 25% heavy
        level = random.randint(0, 3)
        if level == 0:
            return img
        params = get_params(level, seed=None)
        return apply_martian_dust_storm(img, params, seed=None)

    def _training_transform(self, img, mask):

        #Random resize, 0.5x-2.0x
        scale = random.uniform(0.5, 2.0)
        width, height = img.size
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        img = img.resize((resized_width, resized_height),Image.Resampling.BILINEAR)
        mask = mask.resize((resized_width, resized_height),Image.Resampling.NEAREST)

        #50% chance of Horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # Ensure a full 512x512 crop is possible after the random resize.
        img, mask = self._pad_to_crop_size(img, mask)

        #Random 512x512 crop
        width, height = img.size
        left = random.randint(0, width - self.size)
        top = random.randint(0, height - self.size)
        crop_box = (left, top, left + self.size, top + self.size)
        img = img.crop(crop_box)
        mask = mask.crop(crop_box)

        return img, mask

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.labels[idx])
        mask = mask.point(lambda v: self.unknown_class if v == 255 else v)

        if self.training:
            if self.distortion:
                img = self._dust(img)
            img, mask = self._training_transform(img, mask)

        #if the image is not being used for training, we split it into four non-overlapping 
        #512x512 quadrants for validation.
        else:
            crop_boxes = [
                (0, 0, self.size, self.size),                        #top-left
                (self.size, 0, 2 * self.size, self.size),            #top-right
                (0, self.size, self.size, 2 * self.size),            #bottom-left
                (self.size, self.size, 2 * self.size, 2 * self.size) #bottom-right
            ]

            image_parts = []
            mask_parts = []

            for crop_box in crop_boxes:
                img_part = img.crop(crop_box)
                mask_part = mask.crop(crop_box)

                img_part = (np.asarray(img_part, dtype=np.float32) / 255. - MEAN) / STD
                mask_part = np.asarray(mask_part, dtype=np.int64)

                image_parts.append(torch.from_numpy(img_part.copy()).permute(2, 0, 1))
                mask_parts.append(torch.from_numpy(mask_part.copy()))

            # image_parts: [4, 3, 512, 512]
            # mask_parts:  [4, 512, 512]
            return (torch.stack(image_parts), torch.stack(mask_parts))
            

            #(OLD) if we aren't traning, take a 512 x 512 center crop of the image and mask.
            # width, height = img.size

            # left = (width - self.size) // 2
            # top = (height - self.size) // 2
            # right = left + self.size
            # bottom = top + self.size

            # img = img.crop((left, top, right, bottom))
            # mask = mask.crop((left, top, right, bottom))


        img = (np.asarray(img, dtype=np.float32) / 255. - MEAN) / STD
        mask = np.asarray(mask, dtype=np.int64)

        #From ChatGPT: .copy() makes the arrays writable/contiguous even after PIL operations
        return (torch.from_numpy(img.copy()).permute(2, 0, 1), torch.from_numpy(mask.copy()))