import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from deepalign_fibsem.data.dataset import FIBSEMSslDataset
from deepalign_fibsem.models.unet import UNetDenoise


def train_ssl(data_dir: str, out_ckpt: str, batch_size: int = 8, epochs: int = 10, lr: float = 1e-3, device: str | None = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = FIBSEMSslDataset(data_dir, patch_size=192, patches_per_image=64)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    net = UNetDenoise().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    crit = nn.L1Loss()

    net.train()
    for ep in range(epochs):
        running = 0.0
        pbar = tqdm(dl, desc=f"SSL Epoch {ep+1}/{epochs}")
        for corrupt, clean in pbar:
            corrupt = corrupt.to(device)
            clean = clean.to(device)
            opt.zero_grad()
            denoised, _ = net(corrupt)
            loss = crit(denoised, clean)
            loss.backward()
            opt.step()
            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        running /= max(1, len(dl))
        print(f"[SSL] Epoch {ep+1}/{epochs} - L1: {running:.4f}")
    os.makedirs(os.path.dirname(out_ckpt) or ".", exist_ok=True)
    torch.save({"state_dict": net.state_dict()}, out_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing clean EM images (we synthesize artifacts)")
    parser.add_argument("--out", default="checkpoints/ssl_unet.pt")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train_ssl(args.data_dir, args.out, args.batch, args.epochs, args.lr)
