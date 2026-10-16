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
    def __init__(self, dim, num_heads=2, qkv_bias=True, 
                qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2*dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1: # the R from the paper - how much we will shrink the K and V values
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x):
        # takes x (B, C, H*W) -> returns o (B, C, H*W)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # split q between heads- swap token and head dims

        if self.sr_ratio > 1: # shrink before computing K, V
            # convert to grid
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            
            kv = self.kv(x_) # [B, 256, 2C]
            k, v = kv.chunk(2, dim=-1) # each [B, 256, C]  (first half / second half)
            k = k.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2) # [B, heads, 256, d]
            v = v.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        else:
            kv = self.kv(x) # [B, 256, 2C]
            k, v = kv.chunk(2, dim=-1) # each [B, 256, C]  (first half / second half)
            k = k.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2) # [B, heads, 256, d]
            v = v.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        e = (q @ k.transpose(-2, -1)) * self.scale 
        a = e.softmax(dim=-1)
        a = self.attn_drop(a) 

        o = (a @ v).transpose(1, 2).reshape(B, N, C)
        o = self.proj(o)
        o = self.attn_drop(o)

        return o


class Block(nn.Module):
    """
    Transformer block with self attention and mix-FFN
    """
    def __init__(self):
         self.attn = EffAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        pass

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))

        return x

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
        # drop path rate: drop-path probability per transformer block - stochastic depth - skip residual brach randomly
            # early blocks compute basic features that everything after them depends on, so dropping them would hurt. Later blocks refine those features and are safer to skip.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
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

    