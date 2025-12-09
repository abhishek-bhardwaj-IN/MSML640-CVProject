import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, features: List[int] = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128]
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(2, 2)
        for feat in features:
            self.downs.append(DoubleConv3D(in_channels, feat))
            in_channels = feat

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips

class ArtifactRemovalUNet(nn.Module):
    def __init__(self, features: List[int] = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128]
        self.encoder = UNetEncoder(1, features)
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)
        self.ups = nn.ModuleList()
        for feat in reversed(features):
            self.ups.append(nn.ConvTranspose3d(feat * 2, feat, 2, 2))
            self.ups.append(DoubleConv3D(feat * 2, feat))
        self.final = nn.Conv3d(features[0], 1, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        return torch.sigmoid(self.final(x))

class SpatialTransformer(nn.Module):
    def __init__(self, size: Tuple[int, int, int]):
        super().__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids).unsqueeze(0).float()
        self.register_buffer('grid', grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(3):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode='bilinear')

class ImprovedVoxelMorph(nn.Module):
    def __init__(self, vol_shape: Tuple[int, int, int], features: List[int] = None):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.enc = nn.ModuleList()
        in_ch = 2
        for feat in features:
            self.enc.append(DoubleConv3D(in_ch, feat))
            in_ch = feat
        self.pool = nn.MaxPool3d(2, 2)
        self.dec = nn.ModuleList()
        for i, feat in enumerate(reversed(features[:-1])):
            self.dec.append(nn.ConvTranspose3d(in_ch, feat, 2, 2))
            self.dec.append(DoubleConv3D(feat * 2, feat))
            in_ch = feat
        self.final_up = nn.ConvTranspose3d(in_ch, features[0], 2, 2)
        self.final_conv = DoubleConv3D(features[0] * 2, features[0])
        self.flow = nn.Conv3d(features[0], 3, 3, padding=1)
        self.flow.weight.data.normal_(0, 1e-4)
        self.flow.bias.data.zero_()
        self.stn = SpatialTransformer(vol_shape)

    def forward(self, moving: torch.Tensor, fixed: torch.Tensor):
        x = torch.cat([moving, fixed], dim=1)
        skips = []
        for enc_layer in self.enc:
            x = enc_layer(x)
            skips.append(x)
            x = self.pool(x)
        skips = skips[::-1][1:]
        for i in range(0, len(self.dec), 2):
            x = self.dec[i](x)
            if i // 2 < len(skips):
                skip = skips[i // 2]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            x = self.dec[i + 1](x)
        x = self.final_up(x)
        if x.shape != skips[-1].shape:
            x = F.interpolate(x, size=skips[-1].shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skips[-1]], dim=1)
        x = self.final_conv(x)
        flow = self.flow(x)
        moved = self.stn(moving, flow)
        return moved, flow