#!/usr/bin/env python

#################################
#                               #
#    U-Net Model Definition     #
#    created by Warren Kozak    #
#                               #
#################################

# Imports
import torch
import torch.nn as nn

# So we can use pre-trained ResNet34
from torchvision.models import resnet34, ResNet34_Weights

# TODO: Maybe use nn.Sequential for each decoder/decoder Block?
# TODO: Do I need to use batch normalization after each convolutional layer?

class UNetEncoderBlock(nn.Module):
    '''A class representing one block of two convolutional layers followed by a max pool layer in a U-Net.'''

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.norm2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        features = self.activation(x)
        downsampled = self.downsample(features)
        return features, downsampled

class UNetDecoderBlock(nn.Module):
    '''A class representing one block of two convolutional layers followed by a up-conv layer in a U-Net'''

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.norm2 = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=True)
        self.upsample = nn.ConvTranspose2d(output_channels, output_channels // 2, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x

class UNet(nn.Module):
    '''A class implementing the U-Net model for image segmentation.'''

    def __init__(self, resnet_encoder=False):
        super().__init__()
        self.resnet_encoder = resnet_encoder
        self.skip_ch = [64, 64, 128, 256] if resnet_encoder else [64, 128, 256, 512]

        # If resnet_encoder=True, use the pretrained ResNet34 as the encoder for the U-Net.
        # This should help the model train faster and perform better.
        if self.resnet_encoder:
            resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
            self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            
        else:
            self.encoder1 = UNetEncoderBlock(1, 64)
            self.encoder2 = UNetEncoderBlock(64, 128)
            self.encoder3 = UNetEncoderBlock(128, 256)
            self.encoder4 = UNetEncoderBlock(256, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 2, 2)
        )

        # Decoder layers
        self.decoder1 = UNetDecoderBlock(self.skip_ch[3] + 512, 512)
        self.decoder2 = UNetDecoderBlock(self.skip_ch[2] + 256, 256)
        self.decoder3 = UNetDecoderBlock(self.skip_ch[1] + 128, 128)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(self.skip_ch[0] + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        if resnet_encoder:
            self.up_final = nn.ConvTranspose2d(64, 32, 2, 2)
            self.final = nn.Sequential(
                nn.Conv2d(35, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.output = nn.Conv2d(32, 5, 1)
        else:
            self.output = nn.Conv2d(64, 5, 1)

    def forward(self, x):

        inp = x

        if self.resnet_encoder:
            skip1 = self.init_conv(x)
            downsampled = self.maxpool(skip1)
            skip2 = self.layer1(downsampled)
            skip3 = self.layer2(skip2)
            skip4 = self.layer3(skip3)
            x = self.layer4(skip4)

        else:
            # Run the encoder and save the intermediate outputs
            skip1, x = self.encoder1(x)
            skip2, x = self.encoder2(x)
            skip3, x = self.encoder3(x)
            skip4, x = self.encoder4(x)

        # Run the bottleneck
        x = self.bottleneck(x)

        # Run the decoders and concatenate the skip connections
        x = self.decoder1(torch.cat((skip4, x), dim=1))
        x = self.decoder2(torch.cat((skip3, x), dim=1))
        x = self.decoder3(torch.cat((skip2, x), dim=1))
        x = self.decoder4(torch.cat((skip1, x), dim=1))

        # If using the ResNet34 encoder, we need to upscale one more time
        if self.resnet_encoder:
            x = self.up_final(x)
            x = self.final(torch.cat((inp, x), dim=1))
            out = self.output(x)

        else:
            out = self.output(x)

        # The final output segmentation with 5 classes
        # 0: NULL (no label)
        # 1: soil
        # 2: bedrock
        # 3: sand
        # 4: big rock
        return out

if __name__ == "__main__":
    model = UNet(resnet_encoder=True)
    test = torch.rand(1, 3, 512, 512)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1000000} M")
    output = model(test)
    print(f"Output: {output.shape}")
    model.eval()
    torch.onnx.export(
        model,
        (test,),
        "onnx/resnet34-unet.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False
    )