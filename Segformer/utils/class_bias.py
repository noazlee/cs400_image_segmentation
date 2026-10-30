"""
File created by: Noah Lee
Checks the SegFormer model for class bias on the test set:
    1) Counts ground-truth pixels per class from the test labels
    2) Counts predicted pixels per class from the model
    3) Prints both side by side (pred/GT ratio > 1 = over-predicted, < 1 = under-predicted)
       plus a row-normalized confusion matrix (where each true class's pixels went)
"""

import sys
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

SEG_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SEG_DIR.parent
sys.path.insert(0, str(SEG_DIR))
from dataset import MarsDataset
from model import build_segformer

CKPT = SEG_DIR / "models/best_seg_102526.pth"
DUST_LEVEL = 0 # 0 = clean, 1-4 = dustification_level_0-3

CLASS_NAMES = ["soil", "bedrock", "sand", "big rock", "none"]
NUM_CLASSES = len(CLASS_NAMES)

def load_test_set(level):
    test_set = MarsDataset(REPO_ROOT / "data/non-dusted/test", size=512, training=False)
    if level > 0:
        dust = level - 1
        test_set.images = [REPO_ROOT / "data/dusted/test" / f"dustification_level_{dust}" / f"{p.stem}_dusted{dust}.JPG"
                           for p in test_set.images]
    return test_set

@torch.no_grad()
def count_pixels(model, loader, device):
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for images, masks in loader:
        # dataset returns 4 quadrants per image: [B, 4, 3, 512, 512] -> [B*4, 3, 512, 512]
        images = images.flatten(0, 1).to(device, non_blocking=True)
        preds = torch.argmax(model(images), dim=1).cpu().numpy()
        masks = masks.flatten(0, 1).numpy()

        # skip padding (255); "none" (4) is kept so we can see if the model falls back to it
        valid = masks != 255
        gt, pred = masks[valid], preds[valid]
        confusion += np.bincount(gt * NUM_CLASSES + pred, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)

    return confusion

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_segformer(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CKPT, map_location=device))
model = model.to(device).eval()

test_set = load_test_set(DUST_LEVEL)
loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
print(f"{len(test_set)} test images, dust level {DUST_LEVEL}")

confusion = count_pixels(model, loader, device)
gt_counts = confusion.sum(axis=1) # rows = true class
pred_counts = confusion.sum(axis=0) # cols = predicted class
total = gt_counts.sum()

rows = []
for c, name in enumerate(CLASS_NAMES):
    rows.append({
        "class": name,
        "gt_pixels": int(gt_counts[c]),
        "gt_pct": 100 * gt_counts[c] / total,
        "pred_pixels": int(pred_counts[c]),
        "pred_pct": 100 * pred_counts[c] / total,
        "pred_gt_ratio": pred_counts[c] / gt_counts[c] if gt_counts[c] else float("nan"),
        "recall": confusion[c, c] / gt_counts[c] if gt_counts[c] else float("nan"),
    })

print()
for r in rows:
    print(r["class"])
    print(f"GT pixels: {r['gt_pixels']} ({round(r['gt_pct'], 2)}%)")
    print(f"Pred pixels: {r['pred_pixels']} ({round(r['pred_pct'], 2)}%)")
    print(f"Pred/GT: {round(r['pred_gt_ratio'], 2)}")
    print(f"Recall: {round(r['recall'], 3)}")

# each row: where the pixels of that true class ended up (rows sum to 1)
norm = confusion / np.maximum(gt_counts[:, None], 1)
print("\nconfusion (row = true, col = pred, row-normalized)")
print(f"columns: {CLASS_NAMES}")
for name, row in zip(CLASS_NAMES, norm):
    print(f"{name}: {np.round(row, 3)}")

out_dir = SEG_DIR / "results"
out_dir.mkdir(exist_ok=True)

with open(out_dir / f"class_bias_dust{DUST_LEVEL}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

