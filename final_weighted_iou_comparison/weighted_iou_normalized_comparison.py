"""
Compare SegFormer and U-Net mean weighted IoU relative to the expected
random-prediction weighted IoU across dustification levels.
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parent
SEGFORMER_CSV = folder / "Seg_metrics_collapsed(dust).csv"
UNET_CSV = folder / "Unet_metrics_collapsed(dust).csv"
OUTPUT_PATH = folder / "weighted_iou_normalized_comparison(dust).png"

COLUMNS = ["weighted_iou_clean", "weighted_iou_light", "weighted_iou_medium", "weighted_iou_heavy"]
DISPLAY_LABELS = ["Original", "Light Dust", "Medium Dust", "Heavy Dust"]

# Expected weighted IoU for independent, frequency-matched random predictions.
CLASS_PROPORTIONS = np.array([0.20658, 0.27078, 0.06877, 0.00465], dtype=float)
CLASS_PROPORTIONS /= np.sum(CLASS_PROPORTIONS)
weights = 1 / CLASS_PROPORTIONS
weights /= np.sum(weights)
expected_ious = CLASS_PROPORTIONS / (2 - CLASS_PROPORTIONS)
expected_weighted_iou = np.sum(weights * expected_ious)


def load_weighted_iou_csv(path):
    data = {column: [] for column in COLUMNS}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column in COLUMNS:
                value = row[column].strip()
                if value:
                    number = float(value)
                    if np.isfinite(number):
                        data[column].append(number)
    return {column: np.asarray(values, dtype=float) for column, values in data.items()}


def plot_normalized_means(segformer_data, unet_data):
    x = np.arange(1, len(COLUMNS) + 1)
    segformer_means = np.array([np.mean(segformer_data[c]) for c in COLUMNS])
    unet_means = np.array([np.mean(unet_data[c]) for c in COLUMNS])

    if not np.all(np.isfinite(segformer_means)) or not np.all(np.isfinite(unet_means)):
        raise ValueError("One or more dustification levels contain no valid scores.")

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35

    ax.bar(x - bar_width/2, segformer_means / expected_weighted_iou,
        width=bar_width, color="C0", label="SegFormer")

    ax.bar(x + bar_width/2, unet_means / expected_weighted_iou,
        width=bar_width, color="C1", label="U-Net")
    ax.axhline(1, color="gray", linestyle="--", linewidth=1.5,
               label="Random-Prediction Baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_LABELS)
    ax.set_xlabel("Dustification Level", fontweight="bold")
    ax.set_ylabel(r"$M[\mathrm{mWIoU}]\,/\,E[\mathrm{mWIoU}]$", fontweight="bold")
    ax.set_title("SegFormer vs. U-Net Performance Relative to Random Baseline\n(Trained on Dustified Images)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Expected weighted IoU: {expected_weighted_iou:.6f}")
    segformer_data = load_weighted_iou_csv(SEGFORMER_CSV)
    unet_data = load_weighted_iou_csv(UNET_CSV)
    plot_normalized_means(segformer_data, unet_data)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()