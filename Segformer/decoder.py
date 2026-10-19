"""
File created by: Noah Lee
The decoder head of SegFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):     

    def forward(self, x):
        


class MLPDecoder(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, features):
        # features: [c1, c2, c3, c4] -> logits (B, num_classes, H/4, W/4)
        
