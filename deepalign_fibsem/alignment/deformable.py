from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from ..models.registration_net import SimpleRegNet, warp
from ..features.extractor import FeatureExtractor


def gradient_smoothness(flow: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    return dx + dy


def train_deformable(
    dataloader,
    extractor: FeatureExtractor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 10,
    lr: float = 1e-3,
    smooth_weight: float = 0.1,
) -> Dict:
    """
    Train the deformable registration network using feature-space loss.
    dataloader yields (fixed, moving) tensors [B,1,H,W] in [0,1].
    """
    device = torch.device(device)
    net = SimpleRegNet(in_ch=2).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    mse = nn.MSELoss()

    history = {"loss": []}
    extractor.model.eval()
    net.train()
    for ep in range(epochs):
        running = 0.0
        for fixed, moving in dataloader:
            fixed = fixed.to(device)
            moving = moving.to(device)
            with torch.no_grad():
                f_feat = extractor.extract(fixed)
                m_feat = extractor.extract(moving)
            # resize features to same spatial dims if needed (should already match at H/8,W/8)
            # Concatenate features along channel dimension? We feed fixed and moving feature magnitudes as 2-channel input
            f_mag = torch.linalg.vector_norm(f_feat, dim=1, keepdim=True)
            m_mag = torch.linalg.vector_norm(m_feat, dim=1, keepdim=True)
            x = torch.cat([f_mag, m_mag], dim=1)

            opt.zero_grad()
            flow = net(x)
            # Warp moving features using flow upsampled to input size
            # Our net predicts flow at full resolution (because inputs are H/8,W/8); it's okay.
            m_warp = warp(m_mag, flow)
            loss_feat = mse(m_warp, f_mag)
            loss_smooth = gradient_smoothness(flow)
            loss = loss_feat + smooth_weight * loss_smooth
            loss.backward()
            opt.step()
            running += loss.item()
        running /= max(1, len(dataloader))
        history["loss"].append(running)
        print(f"[Deformable] Epoch {ep+1}/{epochs} - loss: {running:.4f}")
    return {"model": net.state_dict(), "history": history}
