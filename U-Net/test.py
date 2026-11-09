"""
Original evaluation script by: Noah Lee (SegFormer). Adapted for the U-Net (ResNet34) model from train.py by Warren Kozak.
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
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from dataset import AI4MarsDataset
from model import UNet

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import metrics

# clean test set. Dusted test images live in DUSTED_DIR/dustification_level_{0-3}/
# as <name>_dusted{N}.JPG and reuse the clean labels
CLEAN_DIR = REPO_ROOT / "data/non-dusted/test"
DUSTED_DIR = REPO_ROOT / "data/dusted/test"

# level 0 = clean, levels 1-4 = dustification_level_0-3 (dust 0 is the lightest dust)
DUST_LABELS = ["clean", "dust 0", "light dust", "medium dust", "heavy dust"]
# levels compared by test_data_dustifications. skipping random dust
DUSTIFICATION_LEVELS = [0, 2, 3, 4]

NUM_CLASSES = 5 # none (unlabelled, 0), soil, bedrock, sand, big rock (same as train.py)

# ImageNet mean/std and val transform, identical to train.py
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
val_transform = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512, fill_mask=0, border_mode=0),
    A.Normalize(MEAN.tolist(), STD.tolist()),
])

# class name -> color, in class id order (0-4). Label 0 (unlabelled) is left see-through
CLASS_COLORS = {
    "none": "#9e9e9e", # gray
    "soil": "#d95f02", # orange
    "bedrock": "#7570b3", # purple
    "sand": "#e6ab02", # yellow
    "big rock": "#1b9e77", # green
}
CLASS_CMAP = ListedColormap(list(CLASS_COLORS.values()))
CLASS_CMAP.set_bad(alpha=0)
OVERLAY_ALPHA = 0.5 # opacity of label/prediction colors drawn over the image

# metric -> display name for the dust-level comparison plot
METRIC_NAMES = {
    "dice": "Dice",
    "rand_index": "Rand Index",
    "iou": "IoU",
    "weighted_iou": "Weighted IoU",
}
# box fill per compared level (DUSTIFICATION_LEVELS order), light -> dark as dust increases
DUST_COLORS = ["#e3eef9", "#e9c99a", "#c98a4b", "#8c5527"]

def load_model(ckpt_path, device):
    model = UNet(resnet_encoder=True)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device).eval()

def load_test_set(level):
    # dusted images reuse the clean test labels; swap the image path in each (image, label) pair
    test_set = AI4MarsDataset(
        str(CLEAN_DIR / "images"),
        str(CLEAN_DIR / "labels"),
        val_transform=val_transform,
    )
    test_set.validation_enabled = True # disable random augmentations (same as validate() in train.py)
    if level > 0:
        dust = level - 1
        test_set.pairs = [
            (DUSTED_DIR / f"dustification_level_{dust}" / f"{img.stem}_dusted{dust}.JPG", label)
            for img, label in test_set.pairs
        ]
    return test_set

def make_loader(test_set):
    return DataLoader(
        test_set,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

def recombine_quadrants(parts):
    #Recombine four 512x512 quadrants into one 1024x1024 image/mask (same as train.py).
    #Input: [B, 4, H, W]
    #Output: [B, 2H, 2W]

    top = torch.cat((parts[:, 0], parts[:, 1]), dim=-1)
    bottom = torch.cat((parts[:, 2], parts[:, 3]), dim=-1)

    return torch.cat((top, bottom), dim=-2)

def predict(model, images):
    # Input: [B, 3, 1024, 1024] images. Runs each 512x512 quadrant through the U-Net independently
    # (same as validate() in train.py). Output: [B, 1024, 1024] class ids
    h, w = images.shape[-2] // 2, images.shape[-1] // 2
    quads = [
        images[:, :, :h, :w],
        images[:, :, :h, w:],
        images[:, :, h:, :w],
        images[:, :, h:, w:],
    ]
    parts = torch.stack([torch.argmax(model(q), dim=1) for q in quads], dim=1)
    return recombine_quadrants(parts)

def mean_scores(scores):
    # per-image scores -> mean of each metric, skipping images where it's undefined (nan)
    return {key: float(np.nanmean(values)) for key, values in scores.items()}

def print_results(scores, name):
    for key, value in mean_scores(scores).items():
        print(f"[{name}] test {METRIC_NAMES[key]} {value:.4f}")

@torch.no_grad()
def evaluate(model, loader, device, level, append=False):
    # Output: per-image scores for each metric (metric -> array, one entry per test image)
    dice_scores = []
    rand_scores = []
    iou_scores = []
    weighted_iou_scores = []

    with open("results-2/dust_metrics.csv", "a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "level", *METRIC_NAMES])
        writer.writeheader()

        for images, masks, names in loader:
            #images: [B, 3, 1024, 1024]
            #masks:  [B, 1024, 1024]
            images = images.to(device, non_blocking=True)

            # [B, 1024, 1024]
            preds = predict(model, images).cpu().numpy()
            masks_np = masks.long().numpy().copy()

            # label 0 = unlabelled -> 255 so it's ignored by the metrics (same as train.py)
            masks_np[masks_np == 0] = 255

            for ground_truth, prediction, name in zip(masks_np, preds, names):
                if (ground_truth != 255).sum() == 0: # no labelled pixels (train.py skips these too)
                    continue

                dice = metrics.dice(ground_truth, prediction, NUM_CLASSES, ignore_id=0)
                rand = metrics.rand_index(ground_truth, prediction)
                iou = metrics.iou(ground_truth, prediction, NUM_CLASSES, ignore_id=0)
                wiou = metrics.weighted_iou(ground_truth, prediction, NUM_CLASSES, ignore_id=0)

                # one row per level, mean of each metric
                writer.writerow({
                    "image": name,
                    "level": DUST_LABELS[level],
                    "dice": dice,
                    "rand_index": rand,
                    "iou": iou,
                    "weighted_iou": wiou,
                })

                dice_scores.append(dice)
                rand_scores.append(rand)
                iou_scores.append(iou)
                weighted_iou_scores.append(wiou)

    return {
        "dice": np.array(dice_scores),
        "rand_index": np.array(rand_scores),
        "iou": np.array(iou_scores),
        "weighted_iou": np.array(weighted_iou_scores),
    }

## Matplotlib code - assisted by Claude
def _to_display(img):
    # undo the ImageNet normalization so the image displays normally.
    # the dataset loads with cv2 (BGR) and the model was trained on BGR, so flip to RGB for display only
    img = np.clip(img.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
    return np.ascontiguousarray(img[..., ::-1])

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

def _predict_sample(model, test_set, idx, device):
    # returns (display image, label with unlabelled masked out, prediction) for one test image
    img, mask, _ = test_set[idx] # img: [3, 1024, 1024], mask: [1024, 1024]
    pred = predict(model, img[None].to(device))[0].cpu().numpy()
    return _to_display(img), np.ma.masked_equal(mask.numpy(), 0), pred

@torch.no_grad()
def viz_predictions(model, test_set, device, out_path, n=10, seed=102326):
    # Input: model, test_dataset
    # Output: saves an n x 3 grid (original | label | prediction) to out_path
    indices = random.Random(seed).sample(range(len(test_set)), n)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    for row, idx in enumerate(indices):
        img_np, mask_np, pred = _predict_sample(model, test_set, idx, device)

        axes[row, 0].imshow(img_np)
        _overlay(axes[row, 1], img_np, mask_np)
        _overlay(axes[row, 2], img_np, pred)
        axes[row, 0].set_ylabel(test_set.pairs[idx][0].stem, fontsize=6)

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
            img_np, mask_np, pred = _predict_sample(model, test_set, idx, device)

            if level == 0:
                _overlay(axes[row, 0], img_np, mask_np)
            _overlay(axes[row, 1 + level], img_np, pred)
        axes[row, 0].set_ylabel(test_sets[0].pairs[idx][0].stem, fontsize=6)

    titles = ["label"] + [f"pred ({DUST_LABELS[level]})" for level in DUSTIFICATION_LEVELS]
    _finish_grid(fig, axes, titles, out_path)

# Claude used for this function
def plot_dust_metrics(all_scores, out_path, title):
    # Input: list of per-image score dicts (from evaluate) ordered clean -> most dust
    # Output: one boxplot panel per metric, one box per dust level, saved to out_path
    labels = [DUST_LABELS[level] for level in DUSTIFICATION_LEVELS]
    x = np.arange(1, len(labels) + 1)
    fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(3.2 * len(METRIC_NAMES), 4.2))

    for ax, (key, name) in zip(axes, METRIC_NAMES.items()):
        # drop images where the metric is undefined (nan)
        data = [scores[key][~np.isnan(scores[key])] for scores in all_scores]
        box = ax.boxplot(
            data, positions=x, widths=0.6, patch_artist=True,
            medianprops=dict(color="#222222", linewidth=1.5),
            whiskerprops=dict(color="#666666"), capprops=dict(color="#666666"),
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#999999",
                            markeredgecolor="none", alpha=0.6),
        )
        for patch, color in zip(box["boxes"], DUST_COLORS):
            patch.set_facecolor(color)
            patch.set_edgecolor("#666666")

        # mean as a labelled white diamond (same value as the printed results / dust_metrics.csv)
        means = [d.mean() for d in data]
        ax.scatter(x, means, marker="D", s=22, color="white", edgecolor="#222222", zorder=3)
        for xi, m in zip(x, means):
            ax.annotate(f"{m:.3f}", (xi, m), textcoords="offset points", xytext=(0, -12),
                        ha="center", fontsize=7, color="#222222",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))

        ax.set_title(name)
        ax.set_xticks(x, [label.replace(" ", "\n") for label in labels])
        ax.set_ylim(top=1.02)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("per-image score")
    fig.suptitle(title)
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
    scores = evaluate(model, test_loader, device)
    print_results(scores, DUST_LABELS[dust_level])

    # Visualize
    os.makedirs("results", exist_ok=True)
    out_path = f"results-2/test_dust{dust_level}.png"
    viz_predictions(model, test_dataset, device, out_path)

    return scores

# runs test_data's steps on the clean set + every dust level, then compares them
def test_data_dustifications(ckpt_path):

    # Load model
    print("Loading model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_path, device)
    os.makedirs("results-2", exist_ok=True)

    # run eval + the standard figure for every level
    test_sets = []
    all_scores = []
    for level in DUSTIFICATION_LEVELS:
        name = DUST_LABELS[level]
        print(f"Loading data ({name})")
        test_dataset = load_test_set(level)
        print(f"{len(test_dataset)} testing images")

        print(f"Evaluating ({name})")
        scores = evaluate(model, make_loader(test_dataset), device, level, append=True)
        print_results(scores, name)
        viz_predictions(model, test_dataset, device, f"results-2/test_dust{level}.png")

        test_sets.append(test_dataset)
        all_scores.append(scores)

    # one row per level, mean of each metric
    with open("results-2/summary_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", *METRIC_NAMES])
        writer.writeheader()
        for level, scores in zip(DUSTIFICATION_LEVELS, all_scores):
            writer.writerow({"level": DUST_LABELS[level], **mean_scores(scores)})

    # Compare results
    plot_dust_metrics(all_scores, "results-2/dust_metrics.png", f"Test metrics by dust level ({Path(ckpt_path).parent.name})")
    viz_predictions_dustifications(model, test_sets, device, "results-2/test_dust_compare.png")

    return all_scores

if __name__ == "__main__":
    ckpt = "output_weighted_loss_resnet34_null_loss/best.pth"
    dust = 0
    DUSTIFICATION = True

    if not DUSTIFICATION:
        test_data(dust, ckpt)
    else:
        test_data_dustifications(ckpt)