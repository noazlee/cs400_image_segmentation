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

# ImageNet mean and standard deviation
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

# Transform for the training set
train_transform = A.Compose([
    A.RandomScale(
        scale_limit=(0.5, 2),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        p=1.0
    ),
    A.PadIfNeeded(min_height=512, min_width=512, fill_mask=0, border_mode=0),
    A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean, std)
])

# Transform for the validation set
val_transform = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512, fill_mask=0, border_mode=0),
    A.Normalize(mean, std)
])

# Initialize the dataset
ai4mars = AI4MarsDataset(
    "/home/kozakw/cs400/cs400_image_segmentation/data/non-dusted/train/images",
    "/home/kozakw/cs400/cs400_image_segmentation/data/non-dusted/train/labels",
    train_transform=train_transform,
    val_transform=val_transform
)

train, val = random_split(
    ai4mars,
    [0.8, 0.2],
    generator=torch.Generator().manual_seed(101926)
)
train_dataloader = DataLoader(train, batch_size=6, shuffle=True)
val_dataloader = DataLoader(val, batch_size=1, shuffle=True)

# Training configuration
num_epochs = 10
lr = 3e-4
weight_decay = 1e-4
log_every = 10
save_dir = "output_version_4"
os.makedirs(save_dir, exist_ok=True)

# Weights & Biases configuration
WANDB_PROJECT = "mars-model-training"
WANDB_ENTITY = "noahzlee7-carleton-college"
wandb.init(
    project=WANDB_PROJECT,
    name=f"[U-Net] Non-Dusty: {lr}, weighted-loss, ResNet34",
    entity=WANDB_ENTITY,
    config={
        "num_epochs": num_epochs,
        "lr": lr,
        "batch_size": 6,
        "img_size": 512,
        "seed": 101926,
    },
)

# Figure out the class distributions in the training set
# Once the count of each pixel label has been created, divide each by the median and
# take the square root to get the final class weight.
print("Calculating relative class weights...")
num_classes = 5
total_counts = torch.zeros(num_classes, dtype=torch.long)

for _, label in train_dataloader:
    label = label.long()
    counts = torch.bincount(label.flatten(), minlength=num_classes)
    total_counts += counts

counts = total_counts.float()
median = counts[1:].median()
class_weights = torch.sqrt(median / counts)
class_weights[0] = 0.0
print(f"Class weights: {class_weights}")

# Initialize the model, loss function, optimizer, and learning rate scheduler
print("Initializing model, loss fn, optimizer...")
model = UNet(resnet_encoder=True).to(device)
loss_fn = nn.CrossEntropyLoss(
    ignore_index=0,
    weight=class_weights.to(device)
)
opt = optim.AdamW(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs * len(train_dataloader))

def recombine_quadrants(parts):
    #Recombine four 512x512 quadrants into one 1024x1024 image/mask.
    #Input: [B, 4, H, W]
    #Output: [B, 2H, 2W]

    top = torch.cat((parts[:, 0], parts[:, 1]), dim=-1)
    bottom = torch.cat((parts[:, 2], parts[:, 3]), dim=-1)

    return torch.cat((top, bottom), dim=-2)

NUM_CLASSES = 5
@torch.no_grad()
def validate(model, dataloader):

    # Disable nondeterministic tranformations
    ai4mars.validation_enabled = True

    model.eval()
    total_loss = 0.0
    n_images = 0
    dice_scores = []
    rand_scores = []
    iou_scores = []
    weighted_iou_scores = []

    for i, data in enumerate(dataloader):
        image, label = data
        image = image.to(device)
        label = label.to(device).long()

        if (label != 0).sum() == 0:
            continue

        predicted_parts = []
        masks = []
        n_loss = 0
        batch_loss = 0.0

        quads = [
            (image[:, :, :512, :512], label[:, :512, :512]),
            (image[:, :, :512, 512:], label[:, :512, 512:]),
            (image[:, :, 512:, :512], label[:, 512:, :512]),
            (image[:, :, 512:, 512:], label[:, 512:, 512:]),
        ]

        # Run each 512x512 quadrant through SegFormer independently
        for img_q, lbl_q in quads:
            #[B, 3, 512, 512]
            logits = model(img_q)
            if (lbl_q != 0).any():
                batch_loss += loss_fn(logits, lbl_q).item()
                n_loss += 1
            predicted_parts.append(torch.argmax(logits, dim=1))
            masks.append(lbl_q)

        # Average loss across the four quadrants
        total_loss += batch_loss / max(n_loss, 1)
        n_images += 1

        # [B, 4, 512, 512]
        predicted_parts = torch.stack(predicted_parts, dim=1)
        masks = torch.stack(masks, dim=1)

        # Reconstruct original 1024x1024 prediction and mask
        # [B, 1024, 1024]
        preds = recombine_quadrants(predicted_parts)
        masks = recombine_quadrants(masks)
        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()

        for ground_truth, prediction in zip(masks_np, preds_np):
            ground_truth = ground_truth.copy()
            ground_truth[ground_truth == 0] = 255
            dice_scores.append(metrics.dice(ground_truth, prediction, NUM_CLASSES))
            rand_scores.append(metrics.rand_index(ground_truth, prediction))
            iou_scores.append(metrics.iou(ground_truth, prediction, NUM_CLASSES))
            weighted_iou_scores.append(metrics.weighted_iou(ground_truth, prediction, NUM_CLASSES))
    
    model.train()

    # Reenable nondeterministic tranformations for training
    ai4mars.validation_enabled = False
        
    return {
        "loss": total_loss / max(n_images, 1),
        "dice": float(np.nanmean(dice_scores)),
        "rand_index": float(np.nanmean(rand_scores)),
        "iou": float(np.nanmean(iou_scores)),
        "weighted_iou": float(np.nanmean(weighted_iou_scores))
    }

global_step = 1
best_val_wiou = float("inf")

print("Beginning training...")
for epoch in range(num_epochs):

    print(f"Epoch {epoch+1}:")
    model.train()

    running_loss = 0.0

    for i, data in enumerate(train_dataloader):
        input, label = data
        input = input.to(device)
        label = label.to(device)

        if (label != 0).sum() == 0:
            continue

        opt.zero_grad()
        output = model(input)
        loss = loss_fn(output, label)
        loss.backward()
        opt.step()
        sched.step()

        running_loss += loss.item()
        if (i + 1) % log_every == 0:
            print(
                f"Batch {i + 1} loss: {running_loss / log_every:.4f}"
            )
            wandb.log({
                "train/loss": running_loss / log_every,
                "train/lr": sched.get_last_lr()[0]
            }, step=global_step)
            running_loss = 0.0

        global_step += 1

    val_metrics = validate(model, val_dataloader)
    print(
        f"val loss {val_metrics['loss']:.4f}"
        f"\nval Dice {val_metrics['dice']:.4f}"
        f"\nval Rand Index {val_metrics['rand_index']:.4f}"
        f"\nval IoU {val_metrics['iou']:.4f}"
    )
    wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

    if val_metrics["weighted_iou"] < best_val_wiou:
        best_val_wiou = val_metrics["weighted_iou"]
        torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
        print(f"Saved new best model to {save_dir}/best.pth")
        wandb.run.summary["best_val_loss"] = best_val_wiou

wandb.finish()
torch.save(model.state_dict(), f'{save_dir}/final.pth')