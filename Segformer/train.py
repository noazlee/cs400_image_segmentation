"""
File created by: Noah Lee
This will be the main training loop for segformer. Will load the data, load the existing model, 
and update weights over time through training.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import MarsDataset
from model import build_segformer

def infinite(loader):
    while True: # restart the loader each time it runs out, so we count iterations not epochs
        yield from loader

@torch.no_grad()
def val_loss(model, loader):
    model.eval()
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        total += compute_loss(model(images), masks).item()
    model.train()
    return total / len(loader)

# Load data
print("Loading data...")
data_path = "data/train/"
mars_dataset = MarsDataset(data_path)
train_dataset, val_dataset = random_split(mars_dataset, [0.8, 0.2],
                                            generator = torch.Generator().manual_seed(101926))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=4, pin_memory=True)
print(f"{len(train_dataset)} training images, \n{len(val_dataset)} validation images")

# Load model
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_segformer(num_classes=4)

# Use pretrained model
pretrained_path = "pretrained/segformer_b0_ade.pth"
state_dict = torch.load(pretrained_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)  # strict=False: linear_pred is new because new classes
print(f"missing keys: {missing}") # should only be decoder.linear_pred.weight/bias
print(f"unexpected keys: {unexpected}") # should be []

model = model.to(device)
print(f"Using {device}")

# Optimizers, Loss
print("Creating optimizer...")
num_iters = 2000
warmup_iters = 50
lr = 6e-5
log_every = 50 # print train loss
val_every = 100 # check val loss, save if best
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

compute_loss = nn.CrossEntropyLoss(ignore_index=255) # ignore unlabelled 255
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=0.01)

# linear warmup, then poly decay (power 1 = straight line) to 0
def lr_lambda(step):
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    return 1 - (step - warmup_iters) / (num_iters - warmup_iters)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop
print("Starting training...")
model.train()
data_iter = infinite(train_loader)
best_val = float("inf")
running_loss = 0.0

for i in range(num_iters):
    images, masks = next(data_iter)
    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True).long()

    logits = model(images)
    loss = compute_loss(logits, masks)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # average train loss since the last print
    running_loss += loss.item()
    if (i + 1) % log_every == 0:
        print(f"iter {i + 1}/{num_iters} \ntrain loss {running_loss / log_every:.4f} \nlr {scheduler.get_last_lr()[0]:.2e}")
        running_loss = 0.0

    if (i + 1) % val_every == 0:
        v = val_loss(model, val_loader)
        print(f"val loss {v:.4f}")
        if v < best_val:
            best_val = v
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            print(f"saved new best to {save_dir}/best.pth")

print(f"Finished training. Best val loss {best_val:.4f}")

# TODO: add metrics to this - isaac
    # Image augmentation
