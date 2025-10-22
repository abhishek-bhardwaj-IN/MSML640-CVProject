import argparse
import os
import numpy as np
import torch
import cv2

from ..data.dataset import imread_gray
from ..features.extractor import FeatureExtractor
from ..alignment.affine import align_with_features
from ..models.registration_net import SimpleRegNet, warp


def load_tensor(path: str) -> torch.Tensor:
    img = imread_gray(path)
    return torch.from_numpy(img[None, None, ...].astype(np.float32))


def run_pipeline(fixed_path: str, moving_path: str, ssl_ckpt: str | None, deform_ckpt: str | None, out_aligned_path: str | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fixed_t = load_tensor(fixed_path)
    moving_t = load_tensor(moving_path)

    # Stage 1: load SSL feature extractor
    extractor = FeatureExtractor(ssl_ckpt, device=device)

    # Stage 2: Affine using features
    aligned_affine_np, M, inliers = align_with_features(fixed_t, moving_t, extractor)

    # Prepare tensors for Stage 3
    fixed_feat = extractor.extract(fixed_t.to(device))
    moving_affine_t = torch.from_numpy(aligned_affine_np[None, None, ...].astype(np.float32)).to(device) / 255.0
    moving_feat = extractor.extract(moving_affine_t)
    f_mag = torch.linalg.vector_norm(fixed_feat, dim=1, keepdim=True)
    m_mag = torch.linalg.vector_norm(moving_feat, dim=1, keepdim=True)

    # Stage 3: Deformable refinement
    if deform_ckpt is not None and os.path.exists(deform_ckpt):
        reg = SimpleRegNet(in_ch=2).to(device)
        state = torch.load(deform_ckpt, map_location=device)
        if isinstance(state, dict) and 'model' in state:
            reg.load_state_dict(state['model'])
        else:
            reg.load_state_dict(state)
        reg.eval()
        with torch.no_grad():
            x = torch.cat([f_mag, m_mag], dim=1)
            flow = reg(x)
            m_mag_warp = warp(m_mag, flow)
        # Use flow predicted on features to warp the affine-aligned moving image
        moved = warp(moving_affine_t, flow)
        moved_np = (moved[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    else:
        moved_np = aligned_affine_np

    if out_aligned_path:
        cv2.imwrite(out_aligned_path, moved_np)
    return moved_np, M


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed")
    parser.add_argument("moving")
    parser.add_argument("--ssl_ckpt", default=None, help="Path to trained SSL U-Net checkpoint")
    parser.add_argument("--deform_ckpt", default=None, help="Path to trained deformable model checkpoint")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_pipeline(args.fixed, args.moving, args.ssl_ckpt, args.deform_ckpt, args.out)
