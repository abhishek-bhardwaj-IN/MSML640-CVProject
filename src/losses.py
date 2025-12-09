import torch
import torch.nn as nn
import torch.nn.functional as F
from models import UNetEncoder

class PerceptualLoss(nn.Module):
    def __init__(self, encoder: UNetEncoder):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat_x, _ = self.encoder(x)
            feat_y, _ = self.encoder(y)
        return F.mse_loss(feat_x, feat_y)

def gradient_loss(flow: torch.Tensor) -> torch.Tensor:
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return (torch.mean(dy**2) + torch.mean(dx**2) + torch.mean(dz**2)) / 3.0