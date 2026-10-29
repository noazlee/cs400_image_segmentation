"""
File created by: Noah Lee
Evaluation script that:
    For dusty/ non-dusty images:
        1) Runs a trained model on the non-dusty test set.
        2) Return metrics
        3) Gets 10 random images from test set, visualize prediction - compare against label, original image, prediction - returns image
    With DUSTIFICATION = True, does the above for the clean test set + every dust level, then
    plots the metrics against dust level and a side-by-side figure of predictions at each level.
"""

import os
import sys
import csv
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

# clean test set (images + labels). Dusted test images live in DUSTED_DIR/dustification_level_{0-3}/
# as <name>_dusted{N}.JPG and reuse the clean labels
DATA_PATHS = {
    0: REPO_ROOT / "data/non-dusted/test",
}
DUSTED_DIR = REPO_ROOT / "data/dusted/test"

# level 0 = clean, levels 1-4 = dustification_level_0-3 (dust 0 is the lightest dust)
DUST_LABELS = ["clean", "dust 0", "dust 1", "dust 2", "dust 3"]

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

# metric -> (display name, line color, marker) for the dust-level comparison plot
METRIC_STYLES = {
    "dice": ("Dice", "#2a78d6", "o"),
    "rand_index": ("Rand Index", "#eb6834", "s"),
    "iou": ("IoU", "#1baf7a", "^"),
    "weighted_iou": ("Weighted IoU", "#8a4fd6", "v"),
    "pixel_acc": ("Pixel acc", "#eda100", "D"),
}

def load_model(ckpt_path, device):
    model = build_segformer(num_classes=NUM_CLASSES)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def load_test_set(level):
    # dusted images reuse the clean test labels; swap the image paths, labels stay paired by index
    test_set = MarsDataset(DATA_PATHS[0], size=512, training=False)
    if level > 0:
        dust = level - 1
        test_set.images = [DUSTED_DIR / f"dustification_level_{dust}" / f"{p.stem}_dusted{dust}.JPG"
                           for p in test_set.images]
    return test_set

def make_loader(test_set):
    return DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

def print_results(results, name):
    print(
        f"[{name}] test Dice {results['dice']:.4f}"
        f"\n[{name}] test Rand Index {results['rand_index']:.4f}"
        f"\n[{name}] test IoU {results['iou']:.4f}"
        f"\n[{name}] test Weighted IoU {results['weighted_iou']:.4f}"
        f"\n[{name}] test pixel acc {results['pixel_acc']:.4f}"
    )

@torch.no_grad()
def evaluate(model, loader, device):
    dice_scores = []
    rand_scores = []
    iou_scores = []
    weighted_iou_scores = []
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
            weighted_iou_scores.append(metrics.weighted_iou(ground_truth, prediction, NUM_CLASSES-1))

        # pixel accuracy over labelled pixels only
        valid = masks_np != 255
        correct += np.sum(preds[valid] == masks_np[valid])
        labelled += np.sum(valid)

    return {
        "dice": float(np.nanmean(dice_scores)),
        "rand_index": float(np.nanmean(rand_scores)),
        "iou": float(np.nanmean(iou_scores)),
        "weighted_iou": float(np.nanmean(weighted_iou_scores)),
        "pixel_acc": float(correct / labelled),
    }

## Matplotlib code - assisted by Claude
def _to_display(img):
    # undo the ImageNet normalization so the image displays normally
    return np.clip(img.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)

def _overlay(ax, img_np, seg):
    # draw the image under the label/prediction so they're easier to compare
    ax.imshow(img_np)
    ax.imshow(seg, cmap=CLASS_CMAP, vmin=0, vmax=NUM_CLASSES - 1,
              alpha=OVERLAY_ALPHA, interpolation="nearest")

def _finish_grid(fig, axes, titles, out_path):
    for col, title in enumerate(titles):
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

@torch.no_grad()
def viz_predictions(model, test_set, device, out_path, n=10, seed=102326):
    # Input: model, test_dataset
    # Output: saves an n x 3 grid (original | label | prediction) to out_path
    indices = random.Random(seed).sample(range(len(test_set)), n)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    for row, idx in enumerate(indices):
        img, mask = test_set[idx]
        pred = torch.argmax(model(img[None].to(device)), dim=1)[0].cpu().numpy()

        img_np = _to_display(img)
        mask_np = np.ma.masked_equal(mask.numpy(), 255)

        axes[row, 0].imshow(img_np)
        _overlay(axes[row, 1], img_np, mask_np)
        _overlay(axes[row, 2], img_np, pred)
        axes[row, 0].set_ylabel(test_set.images[idx].stem, fontsize=6)

    _finish_grid(fig, axes, ["image", "label", "prediction"], out_path)

@torch.no_grad()
def viz_predictions_dustifications(model, test_sets, device, out_path, n=10, seed=102326):
    # Input: model, list of test datasets ordered clean -> most dust (same images, same order)
    # Output: saves an n x (1 + levels) grid (label | prediction at each dust level) to out_path,
    # each prediction drawn over the image the model actually saw at that level
    indices = random.Random(seed).sample(range(len(test_sets[0])), n)
    fig, axes = plt.subplots(n, 1 + len(test_sets), figsize=(3 * (1 + len(test_sets)), 3 * n))

    for row, idx in enumerate(indices):
        for level, test_set in enumerate(test_sets):
            img, mask = test_set[idx]
            pred = torch.argmax(model(img[None].to(device)), dim=1)[0].cpu().numpy()
            img_np = _to_display(img)

            if level == 0:
                _overlay(axes[row, 0], img_np, np.ma.masked_equal(mask.numpy(), 255))
            _overlay(axes[row, 1 + level], img_np, pred)
        axes[row, 0].set_ylabel(test_sets[0].images[idx].stem, fontsize=6)

    titles = ["label"] + [f"pred ({name})" for name in DUST_LABELS[:len(test_sets)]]
    _finish_grid(fig, axes, titles, out_path)

def plot_dust_metrics(all_results, out_path, title):
    # Input: list of metric dicts ordered clean -> most dust
    # Output: line plot of every metric against dust level, saved to out_path
    x = np.arange(len(all_results))
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for key, (name, color, marker) in METRIC_STYLES.items():
        values = [r[key] for r in all_results]
        ax.plot(x, values, color=color, linewidth=2, marker=marker, markersize=7, label=name)
        # label the end points only, so the change from clean -> most dust is readable
        for i in (0, len(values) - 1):
            ax.annotate(f"{values[i]:.3f}", (x[i], values[i]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color="#333333")

    ax.set_xticks(x, DUST_LABELS[:len(all_results)])
    ax.set_xlabel("dust level")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# test function, resuse for non-dusty, dusty, more dusty, etc.
def test_data(dust_level, ckpt_path):

    # Load data
    print("Loading data")
    test_dataset = load_test_set(dust_level)
    test_loader = make_loader(test_dataset)

    print(f"{len(test_dataset)} testing images")

    # Load model
    print("Loading model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_path, device)

    # Metrics
    print("Evaluating")
    results = evaluate(model, test_loader, device)
    print_results(results, DUST_LABELS[dust_level])

    # Visualize
    os.makedirs("results", exist_ok=True)
    out_path = f"results/test_dust{dust_level}.png"
    viz_predictions(model, test_dataset, device, out_path)

    return results

# runs test_data's steps on the clean set + every dust level, then compares them
def test_data_dustifications(ckpt_path):

    # Load model
    print("Loading model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_path, device)
    os.makedirs("results", exist_ok=True)

    # run eval + the standard figure for every level
    test_sets = []
    all_results = []
    for level, name in enumerate(DUST_LABELS):
        print(f"Loading data ({name})")
        test_dataset = load_test_set(level)
        print(f"{len(test_dataset)} testing images")

        print(f"Evaluating ({name})")
        results = evaluate(model, make_loader(test_dataset), device)
        print_results(results, name)
        viz_predictions(model, test_dataset, device, f"results/test_dust{level}.png")

        test_sets.append(test_dataset)
        all_results.append(results)

    # one row per level
    with open("results/dust_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", *METRIC_STYLES])
        writer.writeheader()
        for name, results in zip(DUST_LABELS, all_results):
            writer.writerow({"level": name, **results})

    # Compare results
    plot_dust_metrics(all_results, "results/dust_metrics.png", f"Test metrics by dust level ({Path(ckpt_path).stem})")
    viz_predictions_dustifications(model, test_sets, device, "results/test_dust_compare.png")

    return all_results

ckpt = "models/best_seg_102526.pth"
dust = 0
DUSTIFICATION = True

if not DUSTIFICATION:
    test_data(dust,ckpt)
else:
    test_data_dustifications(ckpt)
