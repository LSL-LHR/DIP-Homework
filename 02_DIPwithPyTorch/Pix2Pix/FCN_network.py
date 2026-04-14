import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output RGB channels with values in [-1, 1]
        )

    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        
        ### FILL: encoder-decoder forward pass
        x1 = self.conv1(x)   # (B, 8, H/2, W/2)
        x2 = self.conv2(x1)  # (B, 16, H/4, W/4)
        x3 = self.conv3(x2)  # (B, 32, H/8, W/8)
        x4 = self.conv4(x3)  # (B, 64, H/16, W/16)
        x5 = self.conv5(x4)  # (B, 128, H/32, W/32)
        x6 = self.conv6(x5)  # (B, 256, H/64, W/64)
        
        # Decoder forward pass
        d1 = self.deconv1(x6)  # (B, 128, H/32, W/32)
        d2 = self.deconv2(d1)  # (B, 64, H/16, W/16)
        d3 = self.deconv3(d2)  # (B, 32, H/8, W/8)
        d4 = self.deconv4(d3)  # (B, 16, H/4, W/4)
        d5 = self.deconv5(d4)  # (B, 8, H/2, W/2)
        output = self.deconv6(d5)  # (B, 3, H, W)
        
        return output
    