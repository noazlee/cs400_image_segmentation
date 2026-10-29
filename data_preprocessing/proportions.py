"""
File Create By: Nora Wootten
"""

import os
from PIL import Image

"""
- `0` - soil
- `1` - bedrock
- `2` - sand
- `3` - big rock
- `255` - NULL (no label)
"""

def get_proportions(filename, counts):
  try:
    with Image.open(filename) as img:
      for pixel in img.get_flattened_data():
        counts[pixel] += 1

  except Exception as e:
      print(f'Error opening {filename}: {e}')

  return counts

def math_proportions(counts):
  proportions = {0:0, 1:0, 2:0, 3:0, 255:0}

  pixel_sum = sum(counts.values())
  for i in counts:
    proportions[i] = float(f"{(counts[i] / pixel_sum):.5f}")

  return proportions


if __name__ == "__main__":
  counts = {0:0, 1:0, 2:0, 3:0, 255:0}
  directory = "../dataset/ai4mars-dataset-merged-0.6/msl/ncam/labels/train"
  num_imgs_processed = 0

  # directory = "../dataset/test_images"

  for file in os.listdir(directory):
    if (num_imgs_processed == 0):
      print(f"0 images processed. \n Counts: {counts}")
    filename = os.path.join(directory, file)
    counts = get_proportions(filename, counts)
    num_imgs_processed += 1
    if (num_imgs_processed % 100 == 0):
      print(f"Image number {num_imgs_processed} processed \n Counts: {counts}")
      print(f"Proportions: {(math_proportions(counts))}")

  print("\n")
  print("All images processed.\n")
  print(f"Final counts: {counts}\n")
  print(f"Final proportions: {math_proportions(counts)}")