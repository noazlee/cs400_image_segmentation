"""
File created by: Noah Lee
Segformer model -> combines the hierarchical encoder and the MLP decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import MixVisionTransformer
from decoder import MLPDecoder

class SegFormer(nn.Module):
    def __init__(self, encoder, decoder, upsample=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.upsample = upsample
 
    def forward(self, x):
        features = self.encoder(x) # [c1, c2, c3, c4] 
        # Four feature maps:
            # c1: [B, 32, 128, 128] (1/4 size)
            # c2: [B, 64, 64, 64] (1/8)
            # c3: [B, 160, 32, 32] (1/16)
            # c4: [B, 256, 16, 16] (1/32)

        logits = self.decoder(features) # [B, num_classes, H/4, W/4]

        if self.upsample:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)

        return logits # [B, num_classes, H, W] after upsampling

# params taken from paper's b0 lightweight settings
def build_segformer(num_classes=5, drop_path_rate=0.1,
                    dropout_ratio=0.1, upsample=True):
    encoder = MixVisionTransformer(
        embed_dims=[32,64,160,256],
        depths=[2,2,2,2],
        drop_path_rate=drop_path_rate,
        num_heads=[1, 2, 5, 8],
        sr_ratios=[8, 4, 2, 1], 
        mlp_ratios=[4, 4, 4, 4],
    )

    decoder = MLPDecoder(
        in_channels=[32, 64, 160, 256],
        embed_dim=256,
        num_classes=num_classes,
        dropout_ratio=dropout_ratio,
    )

    return SegFormer(encoder, decoder, upsample=upsample)

if __name__ == "__main__":
    # check param sizes 
    model = build_segformer()
    out = model(torch.randn(2, 3, 512, 512))
    print(out.shape) # expect [2, 5, 512, 512]
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1000000} M") # approx 3.7M for b0