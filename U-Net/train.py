#!/usr/bin/env python

#################################
#                               #
#     U-Net Training Script     #
#    created by Warren Kozak    #
#                               #
#################################

# Imports
import cv2
import os
import sys
import torch
import wandb

import albumentations as A
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import decode_image
from torchvision.transforms import ToPILImage, v2

# Import the dataset and model
from dataset import AI4MarsDataset
from model import UNet

# Import the metrics module
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import metrics

# Ensure we are using the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform for the training set
train_transform = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512, fill_mask=255, border_mode=0),
    A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5)
])

# Transform for the validation set
val_transform = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512, fill_mask=255, border_mode=0),
    A.CenterCrop(height=512, width=512)
])

# Initialize the dataset
ai4mars_train = AI4MarsDataset(
    "/home/kozakw/cs400/cs400_image_segmentation/data/non-train/images",
    "/home/kozakw/cs400/cs400_image_segmentation/data/train/labels",
    transform=train_transform
)

ai4mars_val = AI4MarsDataset(
    "/home/kozakw/cs400/cs400_image_segmentation/data/non-train/images",
    "/home/kozakw/cs400/cs400_image_segmentation/data/train/labels",
    transform=val_transform
)

train, _ = random_split(ai4mars_train, [0.8, 0.2], seed=102526)
_, val = random_split(ai4mars_val, [0.8, 0.2], seed=102526)
train_dataloader = DataLoader(train, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val, batch_size=1, shuffle=True)

# Training configuration
num_epochs = 10
lr = 0.001
log_every = 10
val_every = 50
save_dir = "output"
os.makedirs(save_dir, exist_ok=True)

# Weights & Biases configuration
WANDB_PROJECT = "mars-model-training"
WANDB_ENTITY = "noahzlee7-carleton-college-org"
wandb.init(
    project=WANDB_PROJECT,
    name=f"[U-Net] Non-Dusty2: {lr}",
    entity=WANDB_ENTITY,
    config={
        "num_epochs": num_epochs,
        "lr": lr,
        "batch_size": 4,
        "img_size": 512,
        "seed": 101926,
    },
)

print("Initializing model, loss fn, optimizer...")
model = UNet().to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=255)
opt = optim.Adam(model.parameters(), lr=lr)

NUM_CLASSES = 5
@torch.no_grad
def validate(model, dataloader):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    rand_scores = []
    iou_scores = []

    for i, data in enumerate(dataloader):
        image, label = data
        image = image.to(device)
        label = label.to(device).long()

        logits = model(image)
        total_loss += loss_fn(logits, label).item()

        pred = torch.argmax(logits, dim=1)

        pred_np = pred.cpu().numpy()
        label_np = label.cpu().numpy()

        for ground_truth, prediction in zip(label_np, pred_np):
            ground_truth = ground_truth.copy()
            ground_truth[ground_truth == 0] = 255
            dice_scores.append(metrics.dice(ground_truth, prediction, NUM_CLASSES-1))
            rand_scores.append(metrics.rand_index(ground_truth, prediction))
            iou_scores.append(metrics.iou(ground_truth, prediction, NUM_CLASSES-1))
    
        model.train()
        
        return {
            "loss": total_loss / len(dataloader),
            "dice": float(np.nanmean(dice_scores)),
            "rand_index": float(np.nanmean(rand_scores)),
            "iou": float(np.nanmean(iou_scores)),
        }


print("Beginning training...")
for epoch in range(num_epochs):

    print(f"Epoch {epoch+1}:")
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
        if (i + 1) % log_every == 0:
            print(
                f"Batch {i + 1} loss: {running_loss / 10:.4f}"
            )
            wandb.log({"train/loss": running_loss / log_every})
            running_loss = 0.0

        if (i + 1) % val_every == 0:
            val_metrics = validate(model, val_dataloader)
            print(
                f"val loss {val_metrics['loss']:.4f}"
                f"\nval Dice {val_metrics['dice']:.4f}"
                f"\nval Rand Index {val_metrics['rand_index']:.4f}"
                f"\nval IoU {val_metrics['iou']:.4f}"
            )
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=i + 1)


wandb.finish()
torch.save(model.state_dict(), 'output/unet_weights.pth')