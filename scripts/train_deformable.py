import argparse
import torch
from torch.utils.data import DataLoader

from deepalign_fibsem.data.dataset import PairSliceDataset
from deepalign_fibsem.features.extractor import FeatureExtractor
from deepalign_fibsem.alignment.deformable import train_deformable


def train_def(fixed_list, moving_list, ssl_ckpt: str, out_ckpt: str = "checkpoints/deformable.pt", batch: int = 4, epochs: int = 10, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairSliceDataset(fixed_list, moving_list)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    extractor = FeatureExtractor(ssl_ckpt, device=device)
    result = train_deformable(dl, extractor, device=device, epochs=epochs, lr=lr)
    torch.save(result, out_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed", nargs='+', help="List of fixed image paths")
    parser.add_argument("moving", nargs='+', help="List of moving image paths")
    parser.add_argument("--ssl_ckpt", required=True)
    parser.add_argument("--out", default="checkpoints/deformable.pt")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    assert len(args.fixed) == len(args.moving)
    train_def(args.fixed, args.moving, args.ssl_ckpt, args.out, args.batch, args.epochs, args.lr)
