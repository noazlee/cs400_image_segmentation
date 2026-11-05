"""
Compare SegFormer and U-Net weighted IoU across dustification levels.

File created by: Isaac Lund
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
SEGFORMER_CSV = ROOT / "Segformer" / "results" / "weighted_iou.csv"
UNET_CSV = ROOT / "U-Net" / "results" / "weighted_iou.csv"
OUTPUT_PATH = ROOT / "weighted_iou_comparison.png"

COLUMNS = ["original", "dust1", "dust2", "dust3"]
DISPLAY_LABELS = ["Original", "Dust Level 1", "Dust Level 2", "Dust Level 3"]

def load_weighted_iou_csv(path):
    """
    Returns a dictionary:

    {
        "original": np.array([...]),
        "dust1": np.array([...]),
        "dust2": np.array([...]),
        "dust3": np.array([...])
    }
    """

    data = {column: [] for column in COLUMNS}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            for column in COLUMNS:

                value = row[column]

                # Ignore empty entries
                if value == "":
                    continue

                data[column].append(float(value))

    return {
        column: np.array(values)
        for column, values in data.items()
    }


def plot_weighted_iou(segformer_data, unet_data):

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(1, 5)

    # Move SegFormer slightly left and U-Net slightly right
    offset = 0.17
    segformer_positions = x - offset
    unet_positions = x + offset

    box_width = 0.28

    segformer_values = [
        segformer_data[column]
        for column in COLUMNS
    ]

    unet_values = [
        unet_data[column]
        for column in COLUMNS
    ]

    # SegFormer
    segformer_boxes = ax.boxplot(
        segformer_values,
        positions=segformer_positions,
        widths=box_width,
        patch_artist=True,
        medianprops={"color": "black","linewidth": 1.5},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.5
        }
    )

    # U-Net
    unet_boxes = ax.boxplot(
        unet_values,
        positions=unet_positions,
        widths=box_width,
        patch_artist=True,
        medianprops={"color": "black","linewidth": 1.5},
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "alpha": 0.5
        }
    )

    for box in segformer_boxes["boxes"]:
        box.set_facecolor("C0")
        box.set_alpha(0.65)

    for box in unet_boxes["boxes"]:
        box.set_facecolor("C1")
        box.set_alpha(0.65)

    # Axes
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_LABELS)

    ax.set_xlabel("Dustification Level")
    ax.set_ylabel("Weighted IoU")

    ax.set_ylim(0, 1)

    ax.set_title("SegFormer vs. U-Net Performance Across Dustification Levels")

    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax.set_axisbelow(True)

    # Legend
    legend = [
        Patch(facecolor="C0", alpha=0.65, label="SegFormer"),
        Patch(facecolor="C1", alpha=0.65, label="U-Net")
    ]

    ax.legend(handles=legend)

    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")

    plt.show()


def main():
    segformer_data = load_weighted_iou_csv(SEGFORMER_CSV)
    unet_data = load_weighted_iou_csv(UNET_CSV)

    plot_weighted_iou(segformer_data, unet_data)

if __name__ == "__main__":
    main()