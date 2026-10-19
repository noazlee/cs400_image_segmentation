"""
Written by Claude:
Randomly splits data/mcam_images + data/mcam_labels into train and test sets:
    data/train/images, data/train/labels
    data/test/images,  data/test/labels
Image/label pairs stay together. Files are copied, so mcam_images/mcam_labels are untouched.
Same seed -> same split every time.
Run from the Segformer folder: python utils/split_data.py
"""

import random
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_DIR = DATA_DIR / "mcam_images"
LABEL_DIR = DATA_DIR / "mcam_labels"
TEST_FRACTION = 0.2
SEED = 101926

# pair each image with its label by filename (label = image name + "_15033_merged.png")
labels = {p.name.split("_15033")[0]: p for p in LABEL_DIR.glob("*.png")}  # look up by image name
pairs = []
for image in sorted(IMAGE_DIR.glob("*.JPG")):
    if image.stem in labels:
        pairs.append((image, labels[image.stem]))
    else:
        print(f"no label for {image.name}, skipping")

random.Random(SEED).shuffle(pairs)
n_test = int(len(pairs) * TEST_FRACTION)
splits = {"test": pairs[:n_test], "train": pairs[n_test:]}

for split, split_pairs in splits.items():
    image_out = DATA_DIR / split / "images"
    label_out = DATA_DIR / split / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for image, label in split_pairs:
        shutil.copy(image, image_out / image.name)
        shutil.copy(label, label_out / label.name)

    print(f"{split}: {len(split_pairs)} pairs -> {DATA_DIR / split}")