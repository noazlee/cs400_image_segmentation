'''
File Created by: Noah Lee - borrowing from Nora's

Adds dust to the expert-labelled test images, using the functions from dustification.py.
Reads data/expert/non-dusted/images, saves to data/expert/dusted/dustification_level_{0-3}.
'''

import os
from dustification import get_params, apply_martian_dust_storm

# Run from the folder data_preprocessing
if __name__ == "__main__":
    
    seed = 102326 # Same seed as dustification.py, so the dust levels match the rest of the data
    folder = "../data/expert/non-dusted/images"

    for i in range(4):
        print(f"Dustification level {i}...")
        output_folder = f"../data/expert/dusted/dustification_level_{i}"
        os.makedirs(output_folder, exist_ok=True)

        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(folder, file)

                # Get random params, based on dustification level
                params = get_params(i, seed)

                # Applies the dust storm, saves result as result
                result = apply_martian_dust_storm(filepath, params, seed)

                # Saves with _dusted{i} added to the end of the name, same as dustification.py
                name, extension = os.path.splitext(file)
                result.save(os.path.join(output_folder, name + f"_dusted{i}" + extension))
