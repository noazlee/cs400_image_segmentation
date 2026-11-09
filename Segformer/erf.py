"""
Created by: Noah Lee
Effective receptive field (ERF) of one SegFormer encoder stage, following the
RepLKNet erf/ scripts (https://github.com/DingXiaoH/RepLKNet-pytorch/tree/main/erf).

For each 512x512 test quadrant:
    1) run the encoder and take the center pixel of the chosen stage's feature map
    2) backprop that pixel (summed over channels) to the input image
    3) keep how much each input pixel contributed (relu of the gradient, summed over RGB)
Averages the map over NUM_IMAGES images, then saves a heatmap and prints how much of the
image is needed to cover 20/30/50/99% of the total contribution (bigger -> larger ERF).
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import MarsDataset
from model import build_segformer

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = REPO_ROOT / "data/non-dusted/test"

CKPT = "models/best_seg_103026_mixedloss.pth"
NUM_CLASSES = 5
STAGE = 0 # encoder stage 0-3 (c1-c4); 3 is the deepest, 1/32 resolution
NUM_IMAGES = 20 # each image gives 4 quadrants -> 80 samples


def get_input_grad(model, samples, stage):
    # Input: samples [B, 3, 512, 512] with requires_grad
    # Output: [512, 512] contribution of each input pixel to the stage's center feature
    features = model.encoder(samples)[stage] # [B, C, h, w]
    h, w = features.shape[-2:] # keep the cell in the middle - squish to one number over all channels
    central_point = torch.relu(features[:, :, h // 2, w // 2]).sum()
    grad = torch.autograd.grad(central_point, samples)[0] # d(scalar)/d(every input pixel) - how much each input affected that score
    grad = torch.nn.functional.relu(grad) # only keep pixels that push center feature up
    aggregated = grad.sum((0, 1)) # sum over quadrants + RGB channels -> [512, 512]
    grad_map = aggregated.cpu().numpy()
    return grad_map

def area_ratio(data, thresh):
    # grow a square around the center until it holds > thresh of the total contribution
    # how large a square around the center do you need to capture X% of the total contribution
    h, w = data.shape
    total = data.sum()
    for i in range(1, h // 2):
        area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i].sum()
        if area / total > thresh:
            side = 2 * i + 1
            return side, side * side / (h * w)
    return h, 1.0


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_segformer(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model = model.to(device).eval()

    test_set = MarsDataset(CLEAN_DIR, size=512, training=False)
    erf = np.zeros((512, 512))

    count = 0
    for idx in range(NUM_IMAGES):
        img, _ = test_set[idx] # [4, 3, 512, 512], the 4 quadrants are the batch
        samples = img.to(device).requires_grad_(True)
        grad_map = get_input_grad(model, samples, STAGE)
        if np.isnan(grad_map).any():
            print(f"got NaN on image {idx}, skipping")
            continue
        erf += grad_map
        count += 1
    erf /= count

    # scores span orders of magnitude: log for readability, then rescale to [0, 1]
    erf = np.log10(erf + 1)
    erf = erf / erf.max()

    print(f"stage {STAGE + 1} ERF over {count} images")
    for thresh in [0.2, 0.3, 0.5, 0.99]:
        side, ratio = area_ratio(erf, thresh)
        print(f"  {thresh:.0%} of contribution: {side}x{side} square, {ratio:.3f} of the image")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/erf_stage{STAGE + 1}.png"
    fig, ax = plt.subplots(figsize=(5, 5.4))
    im = ax.imshow(erf, cmap="RdYlGn", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
    ax.set_title(f"ERF, encoder stage {STAGE + 1} (c{STAGE + 1})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")
