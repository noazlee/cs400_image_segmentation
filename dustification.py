import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# Feedback for Gemini:
# - magic numbers like 0.34 should be a const
# - 

def apply_martian_dust_storm(image_path, intensity=0.92):
    """Applies a severe Martian dust storm with smooth sand texture and slightly

    revealed background silhouettes.

    Parameters:
        image_path (str): Path to input image.
        intensity (float): Storm density (0.0 = clear, 1.0 = severe storm).

    Returns:
        PIL.Image: Processed image with soft, heavy haze and visible background elements.
    """
    # 1. Load Image and convert to float array
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    img_np = np.array(img, dtype=np.float32) / 255.0

    # 2. Define Sandy Atmospheric Tone (Warm Ochre-Beige Tint)
    dust_color = np.array([0.84, 0.65, 0.42])

    # 3. Distance/Depth Mask
    y_coords = np.linspace(1.0, 0.0, height)[:, np.newaxis]

    # Exponent ~1.2 keeps solid foreground coverage
    depth_curve = np.power(y_coords, 1.2)

    # Capped at 0.85 (instead of 1.0) to make the background ~1.2x more visible
    fog_factor =  0.34 + depth_curve * (intensity * 0.63)
    fog_factor = np.clip(fog_factor, 0.0, 0.85)[:, :, np.newaxis]
    fog_factor = np.repeat(fog_factor, width, axis=1)

    # 4. Apply Physical Atmospheric Color Scattering - CITE 
    dusty_np = img_np * (1.0 - fog_factor) + dust_color * fog_factor

    # 5. Smoother Procedural Sand Particle Grain (Reduced noise amplitude)
    fine_grain = np.random.normal(0.0, 0.025, (height, width, 3))
    coarse_sand = np.random.normal(0.0, 0.015, (height // 2, width // 2, 3))

    coarse_sand_img = Image.fromarray(
        ((coarse_sand + 0.5) * 255).astype(np.uint8)
    )
    coarse_sand_img = coarse_sand_img.resize(
        (width, height), Image.Resampling.BILINEAR
    )
    coarse_sand = (np.array(coarse_sand_img, dtype=np.float32) / 255.0) - 0.5

    # Reduced particle multiplication for a smoother atmosphere
    procedural_sand = (fine_grain + coarse_sand) * (fog_factor * 0.8)
    dusty_np = np.clip((dusty_np + procedural_sand) * 255.0, 0, 255).astype(
        np.uint8
    )
    dusty_img = Image.fromarray(dusty_np)

    # 6. Toned-Down Distance Blur (Leaves background features discernible)
    max_blur = 18.0 * intensity  # Reduced from 28.0 for 1.2x clearer shapes
    if max_blur > 0.2:
        blurred_img = dusty_img.filter(
            ImageFilter.GaussianBlur(radius=max_blur)
        )

        # Composite blur using fog factor mask
        blur_mask = Image.fromarray(
            (fog_factor[:, :, 0] * 255).astype(np.uint8)
        )
        dusty_img = Image.composite(blurred_img, dusty_img, blur_mask)

    # 7. Restored Contrast for Background Shapes
    contrast_enhancer = ImageEnhance.Contrast(dusty_img)
    dusty_img = contrast_enhancer.enhance(1.0 - (intensity * 0.25))

    return dusty_img


# --- Usage Example ---
if __name__ == "__main__":
    input_filename = "clear_mars_surface.png"
    output_filename = "dusty_mars_surface.jpg"

    print("Applying smoothed dust storm with improved background clarity...")

    result = apply_martian_dust_storm(input_filename, intensity=0.92)
    result.save(output_filename)

    print(f"Done! Saved output to '{output_filename}'")