"""
File created by: Noah Lee
This will be the main training loop for segformer. Will load the data, load the existing model, 
and update weights over time through training.

Updated by Isaac Lund to add validation metrics.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import wandb

from dataset import MarsDataset
from model import build_segformer

# metrics.py and data/ live at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import metrics

NUM_CLASSES = 5 # soil, bedrock, sand, big rock, none (unlabelled 255 -> 4 in dataset.py)

def infinite(loader):
    while True: # restart the loader each time it runs out, so we count iterations not epochs
        yield from loader

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    rand_scores = []
    iou_scores = []

    for images, masks in loader:
        #From ChatGPT: Use non_blocking=True to speed up data transfer to GPU
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        logits = model(images)
        total_loss += compute_loss(logits, masks).item()

        preds = torch.argmax(logits, dim=1)

        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()


        for ground_truth, prediction in zip(masks_np, preds_np):
            # ignore "none" ground-truth pixels so metrics match the 4-class setup
            ground_truth = ground_truth.copy()
            ground_truth[ground_truth == NUM_CLASSES - 1] = 255
            dice_scores.append(metrics.dice(ground_truth, prediction, NUM_CLASSES-1))
            rand_scores.append(metrics.rand_index(ground_truth, prediction))
            iou_scores.append(metrics.iou(ground_truth, prediction, NUM_CLASSES-1))

    model.train()
    
    return {
        "loss": total_loss / len(loader),
        "dice": float(np.nanmean(dice_scores)),
        "rand_index": float(np.nanmean(rand_scores)),
        "iou": float(np.nanmean(iou_scores)),
    }

# Load data
print("Loading data...")
data_path = REPO_ROOT / "data/train"
# We need two MarsDataset objects because training augmentation must be active
# only for the training subset. random_split() normally makes two subsets that
# share the same underlying dataset object, so in that case a single `training` 
# flag would make it impossible to just have training=true for the training subset,
# but not the validation subset.
base_dataset = MarsDataset(data_path, size=512, training=False)

train_split, val_split = random_split(base_dataset, [0.8, 0.2],
        generator=torch.Generator().manual_seed(101926))

train_dataset = Subset(
        MarsDataset(data_path, size=512,training=True, distortion=True),
        train_split.indices)

val_dataset = Subset(
        MarsDataset(data_path, size=512, training=False),
        val_split.indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"{len(train_dataset)} training images, \n{len(val_dataset)} validation images")

# Load model
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_segformer(num_classes=NUM_CLASSES)

# Use pretrained model
pretrained_path = "pretrained/segformer_b0_ade.pth"
state_dict = torch.load(pretrained_path, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False) # strict=False: linear_pred is new because new classes
print(f"missing keys: {missing}") # should only be decoder.linear_pred.weight/bias
print(f"unexpected keys: {unexpected}") # should be []

model = model.to(device)
print(f"Using {device}")

# Optimizers, Loss
print("Creating optimizer...")
num_iters = 6000
warmup_iters = 100
lr = 6e-5
min_lr_ratio = 0.2 # lr decays to lr * min_lr_ratio instead of 0
log_every = 50 # print train loss
val_every = 100 # check val loss, save if best
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

WANDB_PROJECT = "mars-model-training"
WANDB_ENTITY = None

compute_loss = nn.CrossEntropyLoss(ignore_index=255) # ignore unlabelled 255
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay=0.01)

# linear warmup, then poly decay (power 1 = straight line) down to lr * min_lr_ratio
def lr_lambda(step):
    if step < warmup_iters:
        return (step + 1) / warmup_iters
    progress = (step - warmup_iters) / (num_iters - warmup_iters)
    return 1 - (1 - min_lr_ratio) * progress

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

wandb.init(
    project=WANDB_PROJECT,
    name=f"[SEGFORMER] Non-Dusty3: {lr} (Normal)",
    entity=WANDB_ENTITY,
    config={
        "num_iters": num_iters,
        "warmup_iters": warmup_iters,
        "lr": lr,
        "min_lr_ratio": min_lr_ratio,
        "weight_decay": 0.01,
        "batch_size": 8,
        "img_size": 512,
        "seed": 101926,
        "pretrained_path": pretrained_path,
    },
)

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
        wandb.log({"train/loss": running_loss / log_every,
                   "train/lr": scheduler.get_last_lr()[0]}, step=i + 1)
        running_loss = 0.0

    if (i + 1) % val_every == 0:
        val_metrics = validate(model, val_loader)

        print(
            f"val loss {val_metrics['loss']:.4f}"
            f"\nval Dice {val_metrics['dice']:.4f}"
            f"\nval Rand Index {val_metrics['rand_index']:.4f}"
            f"\nval IoU {val_metrics['iou']:.4f}"
        )
        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=i + 1)

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
            print(f"saved new best to {save_dir}/best.pth")
            wandb.run.summary["best_val_loss"] = best_val

print(f"Finished training. Best val loss {best_val:.4f}")
wandb.finish()