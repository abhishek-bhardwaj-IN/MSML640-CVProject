import argparse
import os
import glob
import numpy as np
import cv2
from tqdm import tqdm

from deepalign_fibsem.data.dataset import imread_gray


def random_affine(h, w):
    angle = np.random.uniform(-5, 5)
    scale = np.random.uniform(0.95, 1.05)
    tx = np.random.uniform(-0.05 * w, 0.05 * w)
    ty = np.random.uniform(-0.05 * h, 0.05 * h)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += [tx, ty]
    return M


def make_pairs(input_dir: str, out_dir: str):
    os.makedirs(os.path.join(out_dir, "fixed"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "moving"), exist_ok=True)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(input_dir, e)))
    paths = sorted(paths)
    assert len(paths) > 0, "No images found"
    for p in tqdm(paths, desc="Creating benchmark pairs"):
        img = (imread_gray(p) * 255).astype(np.uint8)
        h, w = img.shape
        # fixed is original; moving is affine-transformed version plus slight noise
        M = random_affine(h, w)
        moving = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        moving = cv2.GaussianBlur(moving, (0, 0), sigmaX=0.6)
        moving = np.clip(moving.astype(np.float32) + np.random.normal(0, 2.0, (h, w)), 0, 255).astype(np.uint8)
        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(out_dir, "fixed", base + "_fixed.png"), img)
        cv2.imwrite(os.path.join(out_dir, "moving", base + "_moving.png"), moving)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory of clean images used to synthesize misaligned pairs")
    parser.add_argument("out_dir", help="Output directory for benchmark pairs")
    args = parser.parse_args()
    make_pairs(args.input_dir, args.out_dir)
