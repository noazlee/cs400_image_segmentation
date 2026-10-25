"""
File created by: Noah Lee
Evaluation script that:
    For dusty/ non-dusty images:
        1) Runs a trained model on the non-dusty test set.
        2) Return metrics
        3) Gets 10 random images from test set, visualize prediction - compare against label, original image, prediction - returns image
"""

import os
import sys
import random
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from dataset import MarsDataset, MEAN, STD
from model import build_segformer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import metrics

# test set for each dust level
DATA_PATHS = {
    0: REPO_ROOT / "data/test",
}

NUM_CLASSES = 5 # soil, bedrock, sand, big rock, none (unlabelled 255 -> 4 in dataset.py)

# class name -> color, in class id order (0-4). Padding 255 is left see-through
CLASS_COLORS = {
    "soil": "#d95f02", # orange
    "bedrock": "#7570b3", # purple
    "sand": "#e6ab02", # yellow
    "big rock": "#1b9e77", # green
    "none": "#9e9e9e", # gray
}
CLASS_CMAP = ListedColormap(list(CLASS_COLORS.values()))
CLASS_CMAP.set_bad(alpha=0)
OVERLAY_ALPHA = 0.5 # opacity of label/prediction colors drawn over the image

def load_model(ckpt_path, device):
    model = build_segformer(num_classes=NUM_CLASSES)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

@torch.no_grad()
def evaluate(model, loader, device):
    dice_scores = []
    rand_scores = []
    iou_scores = []
    correct = 0
    labelled = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        preds = torch.argmax(model(images), dim=1).cpu().numpy()
        masks_np = masks.numpy()

        # ignore "none" ground-truth pixels so metrics match the 4-class setup (same as train.py)
        masks_np = masks_np.copy()
        masks_np[masks_np == NUM_CLASSES - 1] = 255

        for ground_truth, prediction in zip(masks_np, preds):
            dice_scores.append(metrics.dice(ground_truth, prediction, NUM_CLASSES-1))
            rand_scores.append(metrics.rand_index(ground_truth, prediction))
            iou_scores.append(metrics.iou(ground_truth, prediction, NUM_CLASSES-1))

        # pixel accuracy over labelled pixels only
        valid = masks_np != 255
        correct += np.sum(preds[valid] == masks_np[valid])
        labelled += np.sum(valid)

    return {
        "dice": float(np.nanmean(dice_scores)),
        "rand_index": float(np.nanmean(rand_scores)),
        "iou": float(np.nanmean(iou_scores)),
        "pixel_acc": float(correct / labelled),
    }

## Matplotlib code - assisted by Claude
@torch.no_grad()
def viz_predictions(model, test_set, device, out_path, n=10, seed=102326):
    # Input: model, test_dataset
    # Output: saves an n x 3 grid (original | label | prediction) to out_path
    indices = random.Random(seed).sample(range(len(test_set)), n)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    for row, idx in enumerate(indices):
        img, mask = test_set[idx]
        pred = torch.argmax(model(img[None].to(device)), dim=1)[0].cpu().numpy()

        # undo the ImageNet normalization so the image displays normally
        img_np = np.clip(img.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        mask_np = np.ma.masked_equal(mask.numpy(), 255)

        axes[row, 0].imshow(img_np)
        # draw the image under the label and prediction so they're easier to compare
        for col, seg in [(1, mask_np), (2, pred)]:
            axes[row, col].imshow(img_np)
            axes[row, col].imshow(seg, cmap=CLASS_CMAP, vmin=0, vmax=NUM_CLASSES - 1,
                                  alpha=OVERLAY_ALPHA, interpolation="nearest")
        axes[row, 0].set_ylabel(test_set.images[idx].stem, fontsize=6)

    for col, title in enumerate(["image", "label", "prediction"]):
        axes[0, col].set_title(title)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # class color key along the top of the figure
    handles = [Patch(color=color, alpha=OVERLAY_ALPHA, label=name) for name, color in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight())) # leave room for the key
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

# test function, resuse for non-dusty, dusty, more dusty, etc.
def test_data(dust_level, ckpt_path):

    # Load data
    print("Loading data")
    data_path = DATA_PATHS[dust_level]
    test_dataset = MarsDataset(data_path, size=512, training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"{len(test_dataset)} testing images")

    # Load model
    print("Loading model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_path, device)

    # Metrics
    print("Evaluating")
    results = evaluate(model, test_loader, device)
    print(
        f"test Dice {results['dice']:.4f}"
        f"\ntest Rand Index {results['rand_index']:.4f}"
        f"\ntest IoU {results['iou']:.4f}"
        f"\ntest pixel acc {results['pixel_acc']:.4f}"
    )

    # Visualize
    os.makedirs("results", exist_ok=True)
    out_path = f"results/test_dust{dust_level}.png"
    viz_predictions(model, test_dataset, device, out_path)

    return results


ckpt = "models/best_seg_102426_5class.pth"
dust = 0

test_data(dust,ckpt)
