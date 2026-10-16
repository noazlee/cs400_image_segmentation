"""
File created by: Noah Lee
The hierarchical encoder head of SegFormer.
Takes a batch of images and puts it through 4 stages of patch_embedding & transformer_blocks.
Keeps track of feature maps at each stage - returns all 4 groups of feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixFFN(nn.Module):
    """
    MLP with a conv layer
    """
    def __init__(self):
        pass


    def forward(self, x):
        # takes x (B, C, H*W) -> returns o (B, C, H*W)
        pass


class EffAttention(nn.Module):
    """
    Efficient self attention - uses less keys and values
    """
    def __init__(self):
        pass 
    
    def forward(self, x):
        # takes x (B, C, H*W) -> returns o (B, C, H*W)
        pass

class Block(nn.Module):
    """
    Transformer block with self attention and mix-FFN
    """


class OverlapPatchEmbed(nn.Module):
    """
    Convert image/feature map into smaller patches
    """
    def __init__(self, img_size=576, patch_size=7, stride=4, in_chans=3, embed_dim=32):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H = self.W = img_size[0] // patch_size[0]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    # this is taken from the repo exactly - which takes this from PVT, which took it from somewhere else
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d): # kaiming initialization
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # vectors! (B, C, H*W)
        x = self.norm(x)
        return x, H, W

class MixVisionTransformer(nn.Module):
    # defaults are the paper's b0 settings
    def __init__(self, img_size=544, patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2], in_chans=3,
                 embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4), qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=(2, 2, 2, 2), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch embedding
        self.patch_embed1 = OverlapPatchEmbed(...)

        # transformer encoder
        dpr = # i dont get this yet
        cur = 0
        self.block1 = nn.ModuleList([Block(...)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        # ...

    def forward_features(self, x):
        B = x.shape[0]
        out_fms = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # what does this even do?
        out_fms.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out_fms.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out_fms.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out_fms.append(x)

        return out_fms

    def forward(self, x):
        x = self.forward_features(x)
        return x

    