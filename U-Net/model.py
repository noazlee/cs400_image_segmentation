import torch
import torch.nn as nn

# TODO: Maybe use nn.Sequential for each decoder/decoder Block?
# TODO: Do I need to use batch normalization after each convolutional layer?

class UNetEncoderBlock(nn.Module):
    '''A class representing one block of two convolutional layers followed by a max pool layer in a U-Net.'''

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=True)
        self.downsample = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)
        features = self.activation(x)
        downsampled = self.downsample(features)
        return features, downsampled

class UNetDecoderBlock(nn.Module):
    '''A class representing one block of two convolutional layers followed by a up-conv layer in a U-Net'''

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=True)
        self.upsample = nn.ConvTranspose2d(output_channels, output_channels // 2, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.upsample(x)
        return x

class UNet(nn.Module):
    '''A class implementing the U-Net model for image segmentation.'''

    def __init__(self):
        super().__init__()

        # Encoder layers
        self.encoder1 = UNetEncoderBlock(3, 64)
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
        self.decoder1 = UNetDecoderBlock(1024, 512)
        self.decoder2 = UNetDecoderBlock(512, 256)
        self.decoder3 = UNetDecoderBlock(256, 128)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final Convolutional Layer
        self.output = nn.Conv2d(64, 4, 1)

    def forward(self, x):

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

        # The final output segmentation with 4 classes
        return self.output(x)
