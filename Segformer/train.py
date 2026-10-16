"""
File created by: Noah Lee
This will be the main training loop for segformer. Will load the data, load the existing model, 
and update weights over time through training.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

# Load data
print("Loading data...")
data_path = "data/train/"
mars_traindataset = Dataset(data_path, split="train")
mars_valdataset = Dataset(data_path, split="val")
train_loader = DataLoader()
val_loader = DataLoader()

# Load model
print("Loading model...")
model = build_segformer()
load_pretrained(model)

# Optimizers, Loss
print("Creating optimizer...")

# Training loop
print("Starting training...")
model.train()
data_iter = infinite(train_loader)
for i in range(1000):
    images, masks = next(data_iter)
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True).long()

    logits = model(images)
    loss = compute_loss(logits, masks)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()