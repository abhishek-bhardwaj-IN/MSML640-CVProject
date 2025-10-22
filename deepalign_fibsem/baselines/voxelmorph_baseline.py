import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..models.registration_net import SimpleRegNet, warp
from ..data.dataset import PairSliceDataset, imread_gray


def ncc_loss(x, y, window: int = 9):
    # Simplified local NCC for 2D tensors [B,1,H,W]
    pad = window // 2
    filt = torch.ones(1, 1, window, window, device=x.device)
    N = window * window
    sum_x = torch.conv2d(x, filt, padding=pad)
    sum_y = torch.conv2d(y, filt, padding=pad)
    sum_x2 = torch.conv2d(x * x, filt, padding=pad)
    sum_y2 = torch.conv2d(y * y, filt, padding=pad)
    sum_xy = torch.conv2d(x * y, filt, padding=pad)
    u_x = sum_x / N
    u_y = sum_y / N
    cross = sum_xy - u_y * sum_x - u_x * sum_y + N * u_x * u_y
    x_var = sum_x2 - 2 * u_x * sum_x + N * u_x * u_x
    y_var = sum_y2 - 2 * u_y * sum_y + N * u_y * u_y
    ncc = cross * cross / (x_var * y_var + 1e-5)
    return -ncc.mean()


def train_voxelmorph(fixed_paths, moving_paths, device: str = None, epochs: int = 10, lr: float = 1e-3):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairSliceDataset(fixed_paths, moving_paths)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    net = SimpleRegNet(in_ch=2).to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    net.train()
    for ep in range(epochs):
        running = 0.0
        for fixed, moving in dl:
            fixed = fixed.to(device)
            moving = moving.to(device)
            x = torch.cat([fixed, moving], dim=1)
            opt.zero_grad()
            flow = net(x)
            warped = warp(moving, flow)
            loss = ncc_loss(warped, fixed)
            loss.backward()
            opt.step()
            running += loss.item()
        running /= max(1, len(dl))
        print(f"[VoxelMorph] Epoch {ep+1}/{epochs} - NCC loss: {running:.4f}")
    return net.state_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed", nargs='+', help="List of fixed image paths")
    parser.add_argument("moving", nargs='+', help="List of moving image paths")
    args = parser.parse_args()
    assert len(args.fixed) == len(args.moving)
    train_voxelmorph(args.fixed, args.moving)
