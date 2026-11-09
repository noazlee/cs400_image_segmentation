"""
File Created by: Nora Wootten

This file was created for the purpose of saving 8 different images throughout the course of the dustification algorithm.
This file takes in an image, and applies the martian dust storm filter to it, adding iterations of different modifications
to the image throughout the process. Files are saved to a folder figures/dust_storm_parts named in order.
"""

from cs400_image_segmentation.data_preprocessing.dustification import get_params, apply_martian_dust_storm
import os, random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def apply_martian_dust_storm_in_parts(image_path, params, seed):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Unpack Dictionary
    red_tint = params['red_tint']
    green_tint = params['green_tint']
    blue_tint = params['blue_tint']
    gradient_factor = params['gradient_factor']
    baseline_fog = params['baseline_fog']
    intensity = params['intensity']
    variable_fog = params['variable_fog']
    maximum_fog = params['maximum_fog']
    fine_distr = params['fine_distr']
    coarse_distr = params['coarse_distr']
    noise_scaling = params['noise_scaling']
    blur_radius = params['blur_radius']
    final_contrast_factor = params['final_contrast_factor']

    return_images = []

    # 1. Load Image and convert to float array
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    img_np = np.array(img, dtype=np.float32) / 255.0        # Scales image down to a normalized floating point array with values in range [0.0, 1.0]

    # Append next version
    img_1 = img_np
    return_images.append(img_1)

    # 2. Define Sandy Atmospheric Tone (Warm Ochre-Beige Tint)
    dust_color = np.array([red_tint, green_tint, blue_tint])

    # 3. Distance/Depth Mask
    y_coords = np.linspace(1.0, 0.0, height)[:, np.newaxis]     # Creates a 1 dimensional column vector with evenly spaced values, with the number of entries as the height of the image
    depth_curve = np.power(y_coords, gradient_factor)           # Distributes the values in the column vector, y_coords, to be non-linear for the gradient spacing; slightly lowers median values (results in the middle values of the image getting blurrier faster)

    fog_factor = baseline_fog + depth_curve * (intensity * variable_fog)       # Adds fog along gradient
    fog_factor = np.clip(fog_factor, 0.0, maximum_fog)[:, :, np.newaxis]        # Caps the maximum fog level that any pixel in the image can reach, converts the array shape from 2D (height, 1) to 3D (height, 1, 1)
    fog_factor = np.repeat(fog_factor, width, axis=1)                           # Adds fog along the horizontal axis (not just the first column)

    # Append next version
    img_2 = fog_factor[:, :, 0]
    return_images.append(img_2)

    # 4. Apply Physical Atmospheric Color Scattering
    # (Achieved through Linear Alpha Blending, mixing original pixels with dust mask to create illusion of transparency)
    # https://en.wikipedia.org/wiki/Alpha_compositing
    dusty_np = img_np * (1.0 - fog_factor) + dust_color * fog_factor

    # Append next version
    img_3 = dusty_np
    return_images.append(img_3)

    # 5. Smoother Procedural Sand Particle Grain
    # Implemented via zero-center Gaussian Noise
    # geeksforgeeks.org/electronics-engineering/gaussian-noise/
    fine_grain = np.random.normal(0.0, fine_distr, (height, width, 3))      # Generates a high-frequency, pixel-level noise array matching the full resolution of the image
    coarse_sand = np.random.normal(0.0, coarse_distr, (height // 2, width // 2, 3))     # Generates a lower-frequency noise array at half the spatial resolution

    # Standardizes values to ensure the correct values are overlaid with our array
    coarse_sand_img = Image.fromarray(
        ((coarse_sand + 0.5) * 255).astype(np.uint8)
    )
    coarse_sand_img = coarse_sand_img.resize(
        (width, height), Image.Resampling.BILINEAR
    )
    
    # Append next version
    img_4 = coarse_sand_img
    return_images.append(img_4)

    coarse_sand = (np.array(coarse_sand_img, dtype=np.float32) / 255.0) - 0.5

    # Scales the combined grain layers by fog density, and then rescales all values back to [0.0, 255.0]
    procedural_sand = (fine_grain + coarse_sand) * (fog_factor * noise_scaling)
    dusty_np = np.clip((dusty_np + procedural_sand) * 255.0, 0, 255).astype(
        np.uint8
    )
    dusty_img = Image.fromarray(dusty_np)

    # Append next version
    img_5 = dusty_img
    return_images.append(img_5)

    # 6. Toned-Down Distance Blur
    max_blur = blur_radius * intensity
    if max_blur > 0.2:
        blurred_img = dusty_img.filter(
            ImageFilter.GaussianBlur(radius=max_blur)
        )

        # Append next version
        img_6 = blurred_img
        return_images.append(img_6)

        # Composite blur using fog factor mask
        blur_mask = Image.fromarray(
            (fog_factor[:, :, 0] * 255).astype(np.uint8)
        )
        dusty_img = Image.composite(blurred_img, dusty_img, blur_mask) # Blends the blurred image with the sharp image based on blur_mask

        # Append next version
        img_7 = dusty_img
        return_images.append(img_7)

    # 7. Restored Contrast for Background Shapes
    # Enhance final contrast, and smooth contrast for dust storm effect
    contrast_enhancer = ImageEnhance.Contrast(dusty_img)
    dusty_img = contrast_enhancer.enhance(1.0 - (intensity * final_contrast_factor))

    # Append next version
    img_8 = dusty_img
    return_images.append(img_8)

    return return_images


def main():
    seed = 102326
    params = get_params(2,seed)
    results = apply_martian_dust_storm_in_parts("../../dataset/ai4mars-dataset-merged-0.6/msl/mcam/images/1197ML0054560190502966E01_XXXX.JPG", params, seed)
    
    output_folder = "figures/dust_storm_parts"
    os.makedirs(output_folder, exist_ok=True)

    for i, result in enumerate(results):

        # Convert NumPy arrays to PIL images
        if isinstance(result, np.ndarray):
            result = Image.fromarray(
                np.clip(result * 255, 0, 255).astype(np.uint8)
            )

        result.save(
            os.path.join(output_folder, f"dust_storm_{i}.png")
        )

if __name__ == "__main__":
    main()