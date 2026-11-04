'''
File Created by: Nora Wootten

File created to sort the ai4mars dataset images (from folder ai4mars-dataset-merged-0.6/msl/ncam/images/edr)
and labels (from folder ai4mars-dataset-merged-0.6/msl/ncam/labels/train) into train and test sets for the training and
testing of our models. The file checks to ensure that for each image there is a matching label, and if a matching label exists
75% of images/labels are sorted into the train set while the other 25% are sorted into the test set.

# IMPORTANT:
# Before running create the following folder structure:

# data/test/dusty_images,images,labels
# data/train/images,labels
'''



import random, os, shutil

# Run from the folder cs400_image_segmentation
if __name__ == "__main__":
    # Set the seed to ensure reproducibility across ssh/devices
    seed = 102326
    random.seed(seed)

    # Establish folder paths
    input_path_images = "../dataset/ai4mars-dataset-merged-0.6/msl/ncam/images/edr"
    input_path_labels = "../dataset/ai4mars-dataset-merged-0.6/msl/ncam/labels/train"

    # Loop through images, check if the corresponding image is in the labels directory
    # If it is, save 75% of matched image/label pairs to train, and the other 25% to test
    with os.scandir(input_path_images) as images:
        for image in images:
            # Image name is saved as .JPG, labels are the same name.png
            image_name = image.name
            label_name = image.name.replace(".JPG", ".png")

            # Check to make sure there is a matching labeled image (same name)
            if (os.path.exists(f"{input_path_labels}/{label_name}")):
                # Have a 75% bias to save to train over test
                folder_chance = random.choices(["train", "test"], weights=[0.75, 0.25])[0]
                
                # Define the paths to the appropriate folders
                output_path_images = f"../data/non-dusted/{folder_chance}/images"
                output_path_labels = f"../data/non-dusted/{folder_chance}/labels"

                # Duplicate the files and save them to the correct directories
                shutil.copy2(f"{input_path_images}/{image_name}", output_path_images)
                shutil.copy2(f"{input_path_labels}/{label_name}", output_path_labels)
            else:
                # If there is no matched label, continue; a labeled image is necessary for training
                continue