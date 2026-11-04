'''
File Created by: Noah Lee - borrowing from Nora's

File created to sort the expert-labelled ai4mars test set images (from folder ai4mars-dataset-merged-0.6/msl/ncam/images/edr)
and labels (from folder ai4mars-dataset-merged-0.6/msl/ncam/labels/test/masked-gold-*) into data/expert/non-dusted for the
testing of our models.

Labels are saved without the "_merged" in their name (<name>_merged.png -> <name>.png), so images/<name>.JPG and
labels/<name>.png pair by filename in the same way as the rest of the data folder.
'''

import os, shutil

# Which expert labels to use: masked-gold-min{1,2,3}-100agree
# A pixel is only labelled if at least 1 expert agreed on it.
EXPERT_LABELS = "masked-gold-min1-100agree"

# Run from the folder data_preprocessing
if __name__ == "__main__":

    input_path_images = "../dataset/ai4mars-dataset-merged-0.6/msl/ncam/images/edr"
    input_path_labels = f"../dataset/ai4mars-dataset-merged-0.6/msl/ncam/labels/test/{EXPERT_LABELS}"
    output_path_images = "../data/expert/non-dusted/images"
    output_path_labels = "../data/expert/non-dusted/labels"

    os.makedirs(output_path_images, exist_ok=True)
    os.makedirs(output_path_labels, exist_ok=True)

    num_copied = 0

    # Loop through labels
    with os.scandir(input_path_labels) as labels:
        for label in labels:
            # Skip anything that isn't a label
            if not label.name.endswith("_merged.png"):
                continue

            # Labels are saved as <name>_merged.png, images are <name>.JPG
            name = label.name.replace("_merged.png", "")
            image_name = f"{name}.JPG"

            # Check to make sure there is a matching image (same name)
            if (os.path.exists(f"{input_path_images}/{image_name}")):
                # Duplicate the files and save them to the correct directories
                shutil.copy2(f"{input_path_images}/{image_name}", output_path_images)
                shutil.copy2(f"{input_path_labels}/{label.name}", f"{output_path_labels}/{name}.png")
                num_copied += 1
            else:
                # If there is no matched image, continue; a label is no use without its image
                print(f"No image found for {label.name}")
                continue

    print(f"Copied {num_copied} image/label pairs using {EXPERT_LABELS}")
