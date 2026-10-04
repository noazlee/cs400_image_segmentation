import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

"""
Parameters:
    image_path (str): Path to input image.
    params (dict): Overrides for any of DEFAULT_PARAMS (including intensity).
    seed (int): Optional seed so the grain noise is reproducible.

Returns:
    PIL.Image: Processed image with soft, heavy haze and visible background elements.

Function Workflow (described by Gemini):
    1. Load Image ⟶ Convert to RGB float32 array [0.0, 1.0]
    2. Generate Gradient ⟶ Vertical y-coords with exponent 1.2 power curve
    3. Fog Mask ⟶ Compute opacity factor clipped at 0.85
    4. Color Scattering ⟶ Alpha blend image with dust color
    5. Procedural Grain ⟶ Add multi-scale fine/coarse Gaussian noise
    6. Distance Blur ⟶ Composite Gaussian blur via fog alpha mask
    7. Restored Contrast ⟶ Reduce dynamic range by 25% via intensity
"""
DEFAULT_PARAMS = {
    "intensity": 0.92,          # Storm density (0.0 = clear, 1.0 = severe storm)

    # Block 2 (Declare the tint of the sand/dust overlay)
    "red_tint": 0.84,
    "green_tint": 0.65,
    "blue_tint": 0.42,

    # Block 3 (Implement dust/fog overlay)
    "gradient_factor": 1.2,     # Shift factor along the column vector
    "baseline_fog": 0.34,       # Baseline fog overlaid on entire image, the whole iamge has 34% fog intensity
    "variable_fog": 0.63,       # Controls the additional fog along gradient; note summation of baseline_fog + variable_fog = intensity of fog at the top of the image
    "maximum_fog": 0.85,        # Caps the maximum fog level that any pixel in the image can reach

    # Block 5 (Adding graininess/sand to image, Gaussian distribution changes the random amount of the filter a pixel receives...)
    # Creates a more realistic Perlin-like filter by adding fine and coarse layers
    "fine_distr": 0.025,        # standard deviation of normal Gaussian distribution for coarse sand
    "coarse_distr": 0.015,      # standard deviation of normal Gaussian distribution for coarse sand
    "noise_scaling": 0.8,       # scales noise amplitude to scale the harshness of the sand filter

    # Block 6 (Implementing Gaussian Blurring)
    "blur_radius": 18.0,        # Sclaes the maximum blur radius for each pixel

    # Block 7
    "final_contrast_factor": 0.25,  # Reduces and smooths final contrast by 25%
}


def apply_martian_dust_storm(image_path, params=None, seed=None):
    # params overrides any of DEFAULT_PARAMS; seed makes the grain noise reproducible    
    p = {**DEFAULT_PARAMS, **(params or {})}
    intensity = p["intensity"]
    red_tint, green_tint, blue_tint = p["red_tint"], p["green_tint"], p["blue_tint"]
    gradient_factor = p["gradient_factor"]
    baseline_fog = p["baseline_fog"]
    variable_fog = p["variable_fog"]
    maximum_fog = p["maximum_fog"]
    fine_distr = p["fine_distr"]
    coarse_distr = p["coarse_distr"]
    noise_scaling = p["noise_scaling"]
    blur_radius = p["blur_radius"]
    final_contrast_factor = p["final_contrast_factor"]
    if seed is not None:
        np.random.seed(seed)

    # 1. Load Image and convert to float array
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    img_np = np.array(img, dtype=np.float32) / 255.0        # Scales image down to a normalized floating point array with values in range [0.0, 1.0]

    # 2. Define Sandy Atmospheric Tone (Warm Ochre-Beige Tint)
    dust_color = np.array([red_tint, green_tint, blue_tint])

    # 3. Distance/Depth Mask
    y_coords = np.linspace(1.0, 0.0, height)[:, np.newaxis]     # Creates a 1 dimensional column vector with evenly spaced values, with the number of entries as the height of the image
    depth_curve = np.power(y_coords, gradient_factor)           # Distributes the values in the column vector, y_coords, to be non-linear for the gradient spacing; slightly lowers median values (results in the middle values of the image getting blurrier faster)

    fog_factor = baseline_fog + depth_curve * (intensity * variable_fog)       # Adds fog along gradient
    fog_factor = np.clip(fog_factor, 0.0, maximum_fog)[:, :, np.newaxis]        # Caps the maximum fog level that any pixel in the image can reach, converts the array shape from 2D (height, 1) to 3D (height, 1, 1)
    fog_factor = np.repeat(fog_factor, width, axis=1)                           # Adds fog along the horizontal axis (not just the first column)

    # 4. Apply Physical Atmospheric Color Scattering
    # (Achieved through Linear Alpha Blending, mixing original pixels with dust mask to create illusion of transparency)
    # https://en.wikipedia.org/wiki/Alpha_compositing
    dusty_np = img_np * (1.0 - fog_factor) + dust_color * fog_factor

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
    coarse_sand = (np.array(coarse_sand_img, dtype=np.float32) / 255.0) - 0.5

    # Scales the combined grain layers by fog density, and then rescales all values back to [0.0, 255.0]
    procedural_sand = (fine_grain + coarse_sand) * (fog_factor * noise_scaling)
    dusty_np = np.clip((dusty_np + procedural_sand) * 255.0, 0, 255).astype(
        np.uint8
    )
    dusty_img = Image.fromarray(dusty_np)

    # 6. Toned-Down Distance Blur
    max_blur = blur_radius * intensity
    if max_blur > 0.2:
        blurred_img = dusty_img.filter(
            ImageFilter.GaussianBlur(radius=max_blur)
        )

        # Composite blur using fog factor mask
        blur_mask = Image.fromarray(
            (fog_factor[:, :, 0] * 255).astype(np.uint8)
        )
        dusty_img = Image.composite(blurred_img, dusty_img, blur_mask) # Blends the blurred image with the sharp image based on blur_mask

    # 7. Restored Contrast for Background Shapes
    # Enhance final contrast, and smooth contrast for dust storm effect
    contrast_enhancer = ImageEnhance.Contrast(dusty_img)
    dusty_img = contrast_enhancer.enhance(1.0 - (intensity * final_contrast_factor))

    return dusty_img


# --- Usage Example ---
if __name__ == "__main__":
    input_filename = "clear_mars_surface.png"
    output_filename = "dusty_mars_surface.jpg"

    print("Applying smoothed dust storm with improved background clarity...")

    result = apply_martian_dust_storm(input_filename)
    result.save(output_filename)

    print(f"Done! Saved output to '{output_filename}'")