import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetDenoise(nn.Module):
    """
    U-Net for artifact removal on 2D slices.
    Returns both the denoised output and the bottleneck encoder features for downstream alignment.
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.inc = DoubleConv(in_ch, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 4, b * 8))

        self.up1 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.conv1 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.conv2 = DoubleConv(b * 4, b * 2)
        self.up3 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.conv3 = DoubleConv(b * 2, b)
        self.outc = nn.Conv2d(b, 1, 1)

    def encode(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x4 is bottleneck feature map
        return x1, x2, x3, x4

    def forward(self, x):
        x1, x2, x3, x4 = self.encode(x)
        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        u3 = self.conv3(u3)
        out = torch.sigmoid(self.outc(u3))
        return out, x4  # denoised image, encoder bottleneck features
