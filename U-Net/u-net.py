#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import os
import torch

import albumentations as A
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage, v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[ ]:


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

        img = cv2.copyMakeBorder(img, 92, 92, 92, 92, cv2.BORDER_REFLECT_101)

        img = torch.from_numpy(img).permute(2, 0, 1)
        img = v2.functional.to_dtype(img, dtype=torch.float32, scale=True)
        label = torch.from_numpy(label).long()

        return img, label


# In[ ]:


# TODO: Maybe use nn.Sequential for each decoder/decoder Block?
# TODO: Do I need to use batch normalization after each convolutional layer?

class UNetEncoderBlock(nn.Module):
    '''A class representing one block of two convolutional layers followed by a max pool layer in a U-Net.'''

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        features = self.activation(x)
        downsampled = self.downsample(features)
        return features, downsampled

class UNetDecoderBlock(nn.Module):
    '''A class representing one block of two convolutional layers followed by a up-conv layer in a U-Net'''

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3)
        self.activation = nn.ReLU(inplace=True)
        self.upsample = nn.ConvTranspose2d(output_channels, output_channels // 2, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x

class UNet(nn.Module):
    '''A class implementing the U-Net model for image segmentation.'''

    def __init__(self):
        super().__init__()

        # Encoder layers
        self.encoder1 = UNetEncoderBlock(3, 64)
        self.encoder2 = UNetEncoderBlock(64, 128)
        self.encoder3 = UNetEncoderBlock(128, 256)
        self.encoder4 = UNetEncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 2, 2)
        )

        # Decoder layers
        self.decoder1 = UNetDecoderBlock(1024, 512)
        self.decoder2 = UNetDecoderBlock(512, 256)
        self.decoder3 = UNetDecoderBlock(256, 128)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True)
        )

        # Final Convolutional Layer
        self.output = nn.Conv2d(64, 4, 1)

    def forward(self, x):

        # Run the encoder and save the intermediate outputs
        skip1, x = self.encoder1(x)
        skip2, x = self.encoder2(x)
        skip3, x = self.encoder3(x)
        skip4, x = self.encoder4(x)

        # Crop the skip connections to the correct dimensions
        skip1 = skip1[:, :, 88:480, 88:480]
        skip2 = skip2[:, :, 40:240, 40:240]
        skip3 = skip3[:, :, 16:120, 16:120]
        skip4 = skip4[:, :, 4:60, 4:60]

        # Run the bottleneck
        x = self.bottleneck(x)

        # Run the decoders and concatenate the skip connections
        x = self.decoder1(torch.cat((skip4, x), dim=1))
        x = self.decoder2(torch.cat((skip3, x), dim=1))
        x = self.decoder3(torch.cat((skip2, x), dim=1))
        x = self.decoder4(torch.cat((skip1, x), dim=1))

        # The final output segmentation with 4 classes
        return self.output(x)


# In[ ]:


# Transforms for padding and resizing weird shaped data

IMG_TARGET_SIZE = 572
MASK_TARGET_SIZE = 388

img_transform = A.Compose([
        A.LongestMaxSize(max_size=MASK_TARGET_SIZE, mask_interpolation=cv2.INTER_NEAREST),
    A.PadIfNeeded(
        min_height=MASK_TARGET_SIZE,
        min_width=MASK_TARGET_SIZE,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=255
    )
])

# Change the images and labels paths here to match what's on your computer
ai4mars = AI4MarsDataset(
    "/home/warren/School/cs399/cs400_image_segmentation/ai4mars-dataset/msl/mcam/images",
    "/home/warren/School/cs399/cs400_image_segmentation/ai4mars-dataset/msl/mcam/labels/train",
    transform=img_transform
)

train, test = random_split(ai4mars, [0.8, 0.2])
train_dataloader = DataLoader(train, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test, batch_size=1, shuffle=True)


# In[1]:


import numpy as np
from PIL import Image

PALETTE = {0: (140, 102, 64),    # soil
           1: (230, 51, 51),     # bedrock
           2: (242, 217, 77),    # sand
           3: (51, 102, 230),    # big rock
           255: (0, 0, 0)}       # unlabeled

def colorize(label):
    label = np.asarray(label)
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for k, c in PALETTE.items():
        rgb[label == k] = c
    return Image.fromarray(rgb)

def show_sample(img, label):
    image = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    W, H = image.size
    lh, lw = label.shape
    x, y = (W - lw) // 2, (H - lh) // 2          # (572 - 388) // 2 = 92
    box = (x, y, x + lw, y + lh)
    lab = colorize(label)

    label_panel = Image.new("RGB", image.size, (60, 60, 60))   # gray = outside the output
    label_panel.paste(lab, (x, y))

    overlay = image.copy()
    overlay.paste(Image.blend(image.crop(box), lab, 0.5), (x, y))

    canvas = Image.new("RGB", (W * 3, H))
    for i, p in enumerate([image, label_panel, overlay]):
        canvas.paste(p, (i * W, 0))
    return canvas

img, label = ai4mars[150]       # img 3x572x572, label 388x388
show_sample(img, label)  # last expression in a notebook cell displays it


# In[73]:


model = UNet().to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)
opt = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
for epoch in range(num_epochs):

    print(f"EPOCH {epoch+1}:")
    model.train()

    running_loss = 0.0

    for i, data in enumerate(train_dataloader):
        input, label = data
        input = input.to(device)
        label = label.to(device)

        if (label != 255).sum() == 0:
            continue

        opt.zero_grad()
        output = model(input)
        loss = loss_fn(output, label)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"  Batch {i + 1} loss: {running_loss / 10:.4f}")
            running_loss = 0.0

