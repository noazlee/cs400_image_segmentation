"""
File Created by: Nora Wootten

Used to help code WIoU metric, prints the distribution of pixel classes in the dataset.

get_proportions():
Returns a dictionary with keys the class number and values the counts of how many pixels are defined as that class in the
labels in the processes dataset.

math_proportions():
Returns a dictionary with keys the class number and values the percentage of pixels that are labeled as that class.

main():
Iterates through every label in the dataset and processes the pixel counts, occasionally printing the progress. Prints the
final counts of each pixel and the percentage of that pixel in the dataset.
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
  directory = "data/non-dusted/train/labels"
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