"""
File created by: Noah Lee
The decoder head of SegFormer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Simple fully connected layer for unifying channels
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2) # get into vector form (N, H*W, C)
        x = self.proj(x)
        return x
        
class ConvModule(nn.Module):
    """
    1x1 conv layer -> batch normalization layer -> ReLU (not using mmcv ConvModule)
    """
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False) # BN has own bias
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) # save memory using inplace

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)

class MLPDecoder(nn.Module):
    """
    1. Projects to same channels
    2. Upsamples to 1/4 and concats
    3. Fuses channels together
    4. Predicts output mask
    """

    def __init__(self, in_channels=(32, 64, 160, 256), 
                embed_dim=256, num_classes=4, dropout_ratio=0.1):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.linear_c1 = MLP(c1, embed_dim)
        self.linear_c2 = MLP(c2, embed_dim)
        self.linear_c3 = MLP(c3, embed_dim)
        self.linear_c4 = MLP(c4, embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    
    def forward(self, features):
        # features: [c1, c2, c3, c4] -> logits (B, num_classes, H/4, W/4)
        c1, c2, c3, c4 = features
        n = c4.shape[0]
        size = c1.shape[2:]

        # unify channels
        new_c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        new_c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        new_c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        new_c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])

        # upsample
        new_c2 = F.interpolate(new_c2, size=size, mode="bilinear", align_corners=False)
        new_c3 = F.interpolate(new_c3, size=size, mode="bilinear", align_corners=False)
        new_c4 = F.interpolate(new_c4, size=size, mode="bilinear", align_corners=False)

        # concat and fuse 
        c = self.linear_fuse(torch.cat([new_c4, new_c3, new_c2, new_c1], dim=1))
        x = self.dropout(c)
        return self.linear_pred(x)
