"""
File created by: Noah Lee
Remaps AI4Mars labels so classes are 0-3 and unlabeled is 255 (ignored in the loss).
Original: 0 = unlabeled, 1 = soil, 2 = bedrock, 3 = sand, 4 = big rock
New: 255 = unlabeled, 0 = soil, 1 = bedrock, 2 = sand, 3 = big rock
"""

from pathlib import Path
import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

for split in ["train", "test"]:
    src = DATA_DIR / split / "labels"
    dst = DATA_DIR / split / "labels_remapped"
    if not src.exists():
        continue
    dst.mkdir(exist_ok=True)

    paths = sorted(src.glob("*.png"))
    for path in paths:
        label = np.array(Image.open(path))
        new = np.where(label == 0, 255, label - 1).astype(np.uint8)
        Image.fromarray(new).save(dst / path.name)

    print(f"{split}: remapped {len(paths)} labels -> {dst}")
