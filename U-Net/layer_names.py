"""
File created by: Nora Wootten

Tiny function to check UNet layer names for visualizing.py to define the array in variable VISUALIZING_LAYERS.
"""

from model import UNet

model = UNet()

for name, _ in model.named_modules():
    print(name)