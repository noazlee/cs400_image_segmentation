"""
File created by: Nora Wootten

File created to display the feature maps of (row wise) layers of UNet and (column wise) a random channel extracted 
from the layer. Overlays the feature maps on the original image to see the heat map of where the model most reacts
to input images.

File written by Nora Wootten with the assistance of Claude. Comments written by Nora Wootten.
Inspiration taken from Isaac's file visualization.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

from model import UNet

# Access repo root, code pulled from Isaac
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data_preprocessing.dustification import get_params, apply_martian_dust_storm

# Set the paths to the test image and model for easy variable access
IMAGE_PATH = "../data/non-dusted/test/images/NLA_401855249EDR_F0042778NCAM00434M1.JPG"
MODEL_PATH = "output_version_4/best.pth"

# Set image size, number of classes for easy variables access
IMAGE_SIZE = 512
NUM_CLASSES = 5

# MEAN and STD from model
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Layers whose outputs we want to look at (names come from model.named_modules()).
# VISUALIZING_LAYERS = [
#     "layer1.0.conv1", "layer1.2.conv2",
#     "layer2.0.conv1", "layer2.3.conv2",
#     "layer3.0.conv1", "layer3.5.conv2",
#     "layer4.0.conv1", "layer4.2.conv2",
#     "bottleneck.0", "bottleneck.3",
# ]

VISUALIZING_LAYERS = [
    "layer1.0.conv1",
    "layer2.0.conv1",
    "layer3.0.conv1",
    "layer4.0.conv1",
    "bottleneck.0",
]

os.makedirs("viz", exist_ok=True)

# Step 1: Load the trained model
def load_model(device):
    # Build the architecture, load the trained weights, and set eval mode
    model = UNet(resnet_encoder=True).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model

# Helper function to crop the input image to the desired dimensions
def crop(image):
    width, height = image.size
    left = (width - IMAGE_SIZE) // 2
    right = left + IMAGE_SIZE
    top = (height - IMAGE_SIZE) // 2
    bottom = top + IMAGE_SIZE
    image = image.crop((left, top, right, bottom))
    return image

# Step 2: Preprocess the image
def preprocess_image(image):
    # Takes a PIL image and center crops it, size IMAGE_SIZE x IMAGE_SIZE

    # Convert to RGB to match the model input
    image = image.convert("RGB")

    # Center crop to IMAGE_SIZE x IMAGE_SIZE

    image = crop(image)

    # 0->255 becomes 0->1, shape (H, W, 3)
    rgb = np.asarray(image, dtype=np.float32) / 255.0

    # Create tensor for model such that [H, W] -> [1, 1, H, W]
    normalized = (rgb - MEAN) / STD
    tensor = torch.from_numpy(normalized.copy()).permute(2, 0, 1).unsqueeze(0)

    return rgb, tensor

# Load image and preprocess in one step
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess_image(image)


# Step 3: Use hooks to get the layer output/activations
# Assisted by Claude for this step on how to use hooks to best access convolutional layers in UNet
# A hook is a way to view layers without only seeing the input and output of UNet
# Pass in the model, the layer names (listed as VISUALIZING_LAYERS), and activations (empty dict)
def register_hooks(model, layer_names, activations):
    # Yields (name, module) pairs for every submodule in the model
    layers = dict(model.named_modules())

    # Create array to list the hooks so that they can be removed later
    handles = []

    # Function make_hook returns the function hook specific to a given layer
    # Thus, each individual hook produced has unique access to the layer it is embedded in
    # rather than having all hooks refer back to the same variable (non-removable)
    def make_hook(name):
        def hook(module, inputs, output):
            # detach() function only copies values without altering anything
            # i.e. no gradient tracking, only the values
            activations[name] = output.detach()
        return hook

    # For each layer name, check that the name is the module dictionary, return an error otherwise
    # (Note that a module is a collection of layers in UNet, grouped to form a block)
    for name in layer_names:
        if name not in layers:
            raise KeyError(f"Layer '{name}' not found. Use print_layer_names(model) to see valid names.")
        # Append the hook to the handles array
        handles.append(layers[name].register_forward_hook(make_hook(name)))

    # Returns the hook handles array (to remove them later)
    return handles

# Function get_feature_maps takes the model, the image, the activations, and the layer names (VISUALIZING_LAYERS)
def get_feature_maps(model, image, activations, layer_names):
    # Clear any old activations
    activations.clear()

    # Disable gradient tracking with .no_grad()
    with torch.no_grad():
        model(image)
    # Returns an array with the activations for each layer saved in the same order as the layers listed in
    # VISUALIZING_LAYERS. Note that activations are saved as PyTorch tensors.
    return [activations[name].clone() for name in layer_names]


# Step 4: Helder functions for visualizations
# Used Isaac's code for assistance
# mean_activation 
def mean_activation(feature_map):
    # Sets all values in feature map to be absolute value to show the total change in the area, regardless of +/-
    # Sets the activation to be a 2 dimensional 128 x 128 tensor by converting (1, 64, 128, 128) to (128,128)
    # Note, 64 is averaged away, and [0] captures the 1 128x128 image in the tensor entries
    activation = torch.mean(torch.abs(feature_map), dim=1)[0]
    # Moves the image output to the cpu and convers to numpy array
    return activation.cpu().numpy()

# Resizes the images to be the right dimensions for visualization
def upsample_channel(feature_map, channel, output_size=IMAGE_SIZE):
    # Capture 1 channel, while maintaining tensor dimensions
    channel_map = feature_map[:, channel:channel + 1, :, :]
    # Enlarges to 512x512 resolution
    channel_map = F.interpolate(channel_map, size=(output_size, output_size), mode="nearest")
    # Drops batch and channel dimensions, leaves only image size
    channel_map = channel_map[0, 0].cpu().numpy()

    # Normalize this channel to [0, 1] for visualization
    min_val, max_val = channel_map.min(), channel_map.max()
    if max_val > min_val:
        channel_map = (channel_map - min_val) / (max_val - min_val)

    return channel_map

# Pick random channels for each layer, using the same seed to guarantee the same channels for clean and dusty
# Code from Isaac's implementation
def choose_random_channels(feature_maps, channels_per_stage=4, seed=102726):
    rng = np.random.default_rng(seed)
    selected_channels = []
    for feature_map in feature_maps:
        num_channels = feature_map.shape[1]
        channels = rng.choice(num_channels, size=channels_per_stage, replace=False)
        selected_channels.append(channels)
    return selected_channels


# Step 5: Random channels overlaid on the image, create figure plot
# Mostly from Isaac's Code
def visualize_random_channels(image, feature_maps, layer_names, selected_channels,
                              title, save_path, alpha=0.35):
    num_layers = len(feature_maps)
    num_cols = len(selected_channels[0])

    fig = plt.figure(figsize=(14, 4 + 3.2 * num_layers))
    grid = fig.add_gridspec(num_layers + 1, num_cols,
                            height_ratios=[1.3] + [1] * num_layers)

    image_ax = fig.add_subplot(grid[0, :])
    image_ax.imshow(image)
    image_ax.set_title(title, fontsize=16)
    image_ax.axis("off")

    for row, (name, feature_map) in enumerate(zip(layer_names, feature_maps)):
        for col, channel in enumerate(selected_channels[row]):
            ax = fig.add_subplot(grid[row + 1, col])
            ax.imshow(image)

            activation = upsample_channel(feature_map, channel)
            ax.imshow(activation, cmap="viridis", alpha=alpha, interpolation="nearest")

            # Set the title of each image based on the channel and layer name
            ax.set_title(f"{name}, Channel {channel}", fontsize=9)
            ax.axis("off")

    # Adjust the layout of the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # Show the plot
    plt.show()

# Main
def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(device)
    # Created empty dict for activation hooks
    activations = {}
    # Attach hooks, create hook function for each hook, and save each unique function in handles array
    handles = register_hooks(model, VISUALIZING_LAYERS, activations)

    # Load the image, return the original rgb image and the tensor of the image, respectively
    original, image = load_image(IMAGE_PATH)
    # Get the feature maps of each layer
    feature_maps = get_feature_maps(model, image.to(device), activations, VISUALIZING_LAYERS)

    # Generate the dustified version (same as Isaac)
    seed = 102326
    # Apply dust level and seed to get random params for random dustification
    params = get_params(2, seed)
    # Apply dust filer
    dustified_pil = apply_martian_dust_storm(IMAGE_PATH, params, seed)
    # Resize to the center crop of dusty image
    dustified_pil = crop(dustified_pil)

    # Convert dustified_pil to correct array dimensions
    dusty_display, _ = preprocess_image(dustified_pil)

    # Normalize brightness to match the other image (non-dusted)
    brightness_factor = np.mean(original) / np.mean(dusty_display)
    dustified_pil = ImageEnhance.Brightness(dustified_pil).enhance(brightness_factor)

    # Preprocess dusty image and extract feature maps
    dustified_image, dusty_tensor = preprocess_image(dustified_pil)
    dusty_feature_maps = get_feature_maps(model, dusty_tensor.to(device),
                                          activations, VISUALIZING_LAYERS)

    # Hooks are no longer needed, remove them
    for handle in handles:
        handle.remove()

    # Get the same random channels for clean and dusty
    selected_channels = choose_random_channels(feature_maps, channels_per_stage=4, seed=seed)

    # Create the set of images to visualize feature maps
    visualize_random_channels(original, feature_maps, VISUALIZING_LAYERS, selected_channels,
                              "Clean Image", "viz/unet_clean_channels.png")
    visualize_random_channels(dustified_image, dusty_feature_maps, VISUALIZING_LAYERS,
                              selected_channels, "Dustified Image",
                              "viz/unet_dusty_channels.png")


if __name__ == "__main__":
    main()