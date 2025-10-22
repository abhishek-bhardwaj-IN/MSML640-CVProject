import os
import glob
import random
from typing import List, Tuple, Optional

import numpy as np
import cv2
from skimage import io as skio
from skimage.util import view_as_windows
import torch
from torch.utils.data import Dataset


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.ndim == 3 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mi, ma = np.percentile(img, (1, 99))
    if ma - mi < 1e-6:
        mi, ma = img.min(), img.max() if img.max() > img.min() else (0.0, 1.0)
    img = np.clip((img - mi) / (ma - mi + 1e-8), 0.0, 1.0)
    return img


def add_curtaining(img: np.ndarray, strength: float = 0.3, frequency: int = 20) -> np.ndarray:
    h, w = img.shape
    x = np.arange(w, dtype=np.float32)
    # Multiple sine components to simulate curtain bands
    curtain = (np.sin(2 * np.pi * x / (frequency + np.random.randint(-5, 5))) +
               0.5 * np.sin(2 * np.pi * x / (frequency // 2 + np.random.randint(-3, 3))))
    curtain = (curtain - curtain.min()) / (curtain.max() - curtain.min() + 1e-6)
    curtain = curtain * 2 - 1  # [-1, 1]
    curtain = np.tile(curtain[None, :], (h, 1))
    out = img + strength * curtain
    return np.clip(out, 0, 1)


def add_charging(img: np.ndarray, spots: int = 5, max_radius: int = 40, strength: float = 0.6) -> np.ndarray:
    h, w = img.shape
    out = img.copy()
    for _ in range(spots):
        r = np.random.randint(max(10, max_radius // 3), max_radius)
        cy = np.random.randint(r, h - r)
        cx = np.random.randint(r, w - r)
        yy, xx = np.ogrid[:h, :w]
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
        blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.4 * r) ** 2)))
        blob = (blob - blob.min()) / (blob.max() - blob.min() + 1e-8)
        out = np.clip(out + strength * blob * mask, 0, 1)
    return out


def add_gaussian_noise(img: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1)


def add_band_noise(img: np.ndarray, strength: float = 0.2) -> np.ndarray:
    h, w = img.shape
    bands = np.zeros_like(img, dtype=np.float32)
    for y in range(0, h, np.random.randint(20, 60)):
        band_h = np.random.randint(2, 6)
        bands[y:y + band_h, :] = (np.random.rand() * 2 - 1)
    out = img + strength * bands
    return np.clip(out, 0, 1)


def corrupt_image(img: np.ndarray) -> np.ndarray:
    img = _normalize(img)
    # Randomly stack a few artifact types
    if np.random.rand() < 0.9:
        img = add_curtaining(img, strength=np.random.uniform(0.15, 0.4), frequency=np.random.randint(12, 30))
    if np.random.rand() < 0.8:
        img = add_charging(img, spots=np.random.randint(2, 7), max_radius=np.random.randint(20, 60), strength=np.random.uniform(0.3, 0.8))
    if np.random.rand() < 0.7:
        img = add_band_noise(img, strength=np.random.uniform(0.1, 0.3))
    if np.random.rand() < 0.9:
        img = add_gaussian_noise(img, sigma=np.random.uniform(0.01, 0.07))
    return img


def imread_gray(path: str) -> np.ndarray:
    img = skio.imread(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return _normalize(img)


class FIBSEMSslDataset(Dataset):
    """
    Produces pairs (corrupted, clean) from clean images by synthetically corrupting them.
    If images are large, patches are extracted to train denoising/inpainting U-Net.
    """
    def __init__(self, image_dir_or_list: List[str] | str, patch_size: int = 192, patches_per_image: int = 32,
                 random_flip: bool = True):
        if isinstance(image_dir_or_list, str):
            exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
            paths = []
            for e in exts:
                paths.extend(glob.glob(os.path.join(image_dir_or_list, e)))
            self.image_paths = sorted(paths)
        else:
            self.image_paths = list(image_dir_or_list)
        assert len(self.image_paths) > 0, "No images found for SSL dataset."
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.random_flip = random_flip
        self._len = len(self.image_paths) * self.patches_per_image

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        img_idx = idx // self.patches_per_image
        img = imread_gray(self.image_paths[img_idx])
        h, w = img.shape
        ps = self.patch_size
        if h < ps or w < ps:
            # pad
            pad_y = max(0, ps - h)
            pad_x = max(0, ps - w)
            img = np.pad(img, ((0, pad_y), (0, pad_x)), mode='reflect')
            h, w = img.shape
        y = np.random.randint(0, h - ps + 1)
        x = np.random.randint(0, w - ps + 1)
        clean = img[y:y + ps, x:x + ps]
        corrupted = corrupt_image(clean)
        if self.random_flip:
            if np.random.rand() < 0.5:
                clean = np.flip(clean, axis=1)
                corrupted = np.flip(corrupted, axis=1)
            if np.random.rand() < 0.5:
                clean = np.flip(clean, axis=0)
                corrupted = np.flip(corrupted, axis=0)
        clean_t = torch.from_numpy(clean[None, ...].astype(np.float32))
        corrupt_t = torch.from_numpy(corrupted[None, ...].astype(np.float32))
        return corrupt_t, clean_t


class PairSliceDataset(Dataset):
    """
    Returns pairs (fixed, moving) slices for alignment tasks. Can be used for baseline and training registration.
    """
    def __init__(self, fixed_paths: List[str], moving_paths: List[str]):
        assert len(fixed_paths) == len(moving_paths) and len(fixed_paths) > 0
        self.fixed_paths = fixed_paths
        self.moving_paths = moving_paths

    def __len__(self):
        return len(self.fixed_paths)

    def __getitem__(self, idx):
        f = imread_gray(self.fixed_paths[idx])
        m = imread_gray(self.moving_paths[idx])
        f_t = torch.from_numpy(f[None, ...].astype(np.float32))
        m_t = torch.from_numpy(m[None, ...].astype(np.float32))
        return f_t, m_t
