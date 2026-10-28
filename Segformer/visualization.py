"""
File created by: Isaac Lund

This file visualizes intermediate SegFormer encoder feature maps for a single image.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

from model import build_segformer
from dataset import MEAN, STD
from pathlib import Path
import sys

#from ChatGPT: dustification.py lives at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from dustification import get_params, apply_martian_dust_storm

IMAGE_PATH = "../data/non-dusted/test/images/NLA_401855249EDR_F0042778NCAM00434M1.JPG"
MODEL_PATH = "models/best_seg_102526.pth"

IMAGE_SIZE = 512
NUM_CLASSES = 5

def preprocess_image(image):
    width, height = image.size

    left = (width - IMAGE_SIZE) // 2
    top = (height - IMAGE_SIZE) // 2
    right = left + IMAGE_SIZE
    bottom = top + IMAGE_SIZE

    image = image.crop((left, top, right, bottom))

    display_image = np.asarray(image, dtype=np.float32) / 255.0

    normalized = (display_image - MEAN) / STD

    tensor = torch.from_numpy(
        normalized.copy()
    ).permute(2, 0, 1).unsqueeze(0)

    return display_image, tensor

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess_image(image)

def load_model(device):
    model = build_segformer(num_classes=NUM_CLASSES)

    state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model

def get_feature_maps(model, image):
    with torch.no_grad():feature_maps = model.encoder(image)
    return feature_maps

def mean_activation(feature_map):
    activation = torch.mean(torch.abs(feature_map), dim=1)

    # Remove batch dimension
    activation = activation[0]

    return activation.cpu().numpy()

def feature_difference(clean_feature_map, dusty_feature_map):

    difference = torch.abs(dusty_feature_map - clean_feature_map)
    difference = torch.mean(difference, dim=1)

    return difference[0].cpu().numpy()

def upsample_channel(feature_map, channel, output_size=IMAGE_SIZE):
    # Take one channel from a feature map and resize it to the
    # original 512x512 input resolution.
    # feature_map: [1, C, H, W]
    channel_map = feature_map[:, channel:channel + 1, :, :]

    channel_map = F.interpolate(
        channel_map,
        size=(output_size, output_size),
        mode="nearest"
    )

    channel_map = channel_map[0, 0].cpu().numpy()

    # Normalize this channel to [0, 1] for visualization
    min_val = channel_map.min()
    max_val = channel_map.max()

    if max_val > min_val:
        channel_map = (channel_map - min_val) / (max_val - min_val)

    return channel_map


def choose_random_channels(feature_maps, channels_per_stage=4, seed=102726):
    #Choose four random channels from each transformer stage.
    rng = np.random.default_rng(seed)

    selected_channels = []

    for feature_map in feature_maps:
        num_channels = feature_map.shape[1]
        channels = rng.choice(num_channels, size=channels_per_stage, replace=False)
        selected_channels.append(channels)

    return selected_channels


def visualize_random_channels(
    image,
    feature_maps,
    selected_channels,
    title,
    save_path,
    alpha=0.35
):
    #PLOT top: original/dustified image
    #     below:
    #         4 rows = transformer stages
    #         4 columns = random feature channels
    #Each feature channel is resized to 512x512 and overlaid
    #transparently over the input image.

    fig = plt.figure(figsize=(14, 18))

    grid = fig.add_gridspec(5, 4, height_ratios=[1.3, 1, 1, 1, 1])

    # Original image across the full top row
    image_ax = fig.add_subplot(grid[0, :])
    image_ax.imshow(image)
    image_ax.set_title(title, fontsize=16)
    image_ax.axis("off")

    #Four transformer stages
    for stage, feature_map in enumerate(feature_maps):
        #Four random channels per transformer stage
        for column, channel in enumerate(selected_channels[stage]):

            ax = fig.add_subplot(grid[stage + 1, column])
            ax.imshow(image)

            # Feature channel resized to original input resolution
            activation = upsample_channel(feature_map, channel)

            # Transparent feature-map overlay
            ax.imshow(
                activation,
                cmap="viridis",
                alpha=alpha,
                interpolation="nearest"
            )

            ax.set_title(f"Stage {stage + 1}, Channel {channel}")
            ax.axis("off")

        # Label each transformer stage on the left side
        first_ax = fig.axes[1 + stage * 4]

        first_ax.set_ylabel(
            f"Stage {stage + 1}",
            fontsize=14
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def visualize(original, feature_maps, dustified_image, dusty_feature_maps):
    # Original and the four transformer layers
    fig, axes = plt.subplots(3, 5, figsize=(18, 15))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original image")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(dustified_image)
    axes[1, 0].set_title("Dustified image")
    axes[1, 0].axis("off")

    axes[2, 0].text(
        0.5,
        0.5,
        "|Dusty Activation\n− Clean Activation|",
        ha="center",
        va="center",
        fontsize=14
    )
    axes[2, 0].axis("off")

    for i, (feature_map, dusty_feature_map) in enumerate(
        zip(feature_maps, dusty_feature_maps)):

        clean_activation = mean_activation(feature_map)
        dusty_activation = mean_activation(dusty_feature_map)

        difference = feature_difference(feature_map, dusty_feature_map)

        vmin = min(clean_activation.min(), dusty_activation.min())
        vmax = max(clean_activation.max(), dusty_activation.max())

        #Clean
        axes[0, i + 1].imshow(
            clean_activation,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        #Dusty
        axes[1, i + 1].imshow(
            dusty_activation,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax
        )

        #Difference
        axes[2, i + 1].imshow(difference, cmap="viridis")

        axes[0, i + 1].set_title(
            f"Stage {i + 1}\n"
            f"{feature_map.shape[1]} channels, "
            f"{feature_map.shape[2]}×{feature_map.shape[3]}"
        )

        axes[1, i + 1].set_title(f"Stage {i + 1}")
        axes[2, i + 1].set_title(f"Stage {i + 1} Difference")

        for row in range(3):
            axes[row, i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("viz/segformer_fms.png")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original, image = load_image(IMAGE_PATH)
    image = image.to(device)
    model = load_model(device)
    feature_maps = get_feature_maps(model, image)

    seed = 102326
    params = get_params(2, seed)
    dustified_pil = apply_martian_dust_storm(IMAGE_PATH, params, seed)

    #I thought the brightness of the dustification felt a little off, so 
    #I adjusted it to match the original image's brightness. Not sure if this changes anything.
    dusty_array = np.asarray(dustified_pil, dtype=np.float32) / 255.0
    brightness_factor = np.mean(original) /  np.mean(dusty_array)
    dustified_pil = ImageEnhance.Brightness(
        dustified_pil
    ).enhance(brightness_factor)

    dustified_image, dusty_tensor = preprocess_image(dustified_pil)
    dusty_tensor = dusty_tensor.to(device)

    dusty_feature_maps = get_feature_maps(model, dusty_tensor)

    # Choose the same four random channels for the clean
    # and dustified visualizations so they can be compared directly.
    selected_channels = choose_random_channels(
        feature_maps,
        channels_per_stage=4,
        seed=102726
    )

    visualize_random_channels(
        original,
        feature_maps,
        selected_channels,
        "Clean Image",
        "viz/segformer_clean_channels.png"
    )

    visualize_random_channels(
        dustified_image,
        dusty_feature_maps,
        selected_channels,
        "Dustified Image",
        "viz/segformer_dusty_channels.png"
    )

    visualize(original, feature_maps, dustified_image, dusty_feature_maps)

if __name__ == "__main__":
    main()