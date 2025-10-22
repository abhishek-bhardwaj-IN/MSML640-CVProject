import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleRegNet(nn.Module):
    """
    A light U-Net-like model that predicts a 2D displacement field (flow: dx, dy).
    Suitable for VoxelMorph-like experiments on 2D slices.
    """
    def __init__(self, in_ch: int = 2, base_ch: int = 32):
        super().__init__()
        b = base_ch
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(b, b * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(b * 2, b * 4))

        self.up1 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec1 = ConvBlock(b * 4, b * 2)
        self.up2 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec2 = ConvBlock(b * 2, b)
        self.flow = nn.Conv2d(b, 2, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, C=2, H, W] concatenated fixed and moving (or features)
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        u1 = self.up1(c3)
        u1 = torch.cat([u1, c2], dim=1)
        u1 = self.dec1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, c1], dim=1)
        u2 = self.dec2(u2)
        flow = torch.tanh(self.flow(u2))  # small initial displacements in [-1,1]
        return flow


def grid_from_flow(flow: torch.Tensor) -> torch.Tensor:
    """
    Convert pixel displacement flow [B,2,H,W] (dx,dy in pixels) to normalized sampling grid for grid_sample.
    """
    b, _, h, w = flow.shape
    # base grid in normalized coords [-1,1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=flow.device),
        torch.linspace(-1, 1, w, device=flow.device), indexing='ij'
    )
    base_grid = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)
    # Convert flow in pixels to normalized coords
    nx = flow[:, 0] / ((w - 1) / 2)
    ny = flow[:, 1] / ((h - 1) / 2)
    grid = base_grid + torch.stack([nx, ny], dim=-1)
    return grid


def warp(moving: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    grid = grid_from_flow(flow)
    return F.grid_sample(moving, grid, mode='bilinear', padding_mode='border', align_corners=True)
