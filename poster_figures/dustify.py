"""
File Created by: Nora Wootten

This file was created to dustify and save a single image, useful for posters
"""

from cs400_image_segmentation.data_preprocessing.dustification import get_params, apply_martian_dust_storm
import os

seed = 102326
params = get_params(1,seed)
result = apply_martian_dust_storm("../../dataset/ai4mars-dataset-merged-0.6/msl/mcam/images/1197ML0054560190502966E01_XXXX.JPG", params, seed)
output_folder = "figures"
result.save(os.path.join(output_folder, "dusted1.JPG"))