from typing import Tuple, Optional

import numpy as np
import cv2
import torch

from ..features.extractor import FeatureExtractor


def _to_numpy_img(t: torch.Tensor) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if t.ndim == 4:
        t = t[0, 0]
    elif t.ndim == 3:
        t = t[0]
    t = np.clip(t, 0, 1)
    return (t * 255).astype(np.uint8)


def features_to_keyimage(feat: torch.Tensor) -> np.ndarray:
    # feat: [1,C,h,w]
    f = feat.detach().cpu().numpy()
    f = np.linalg.norm(f, axis=1, keepdims=False)  # [1,h,w]
    f = f[0]
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    return (f * 255).astype(np.uint8)


def detect_and_match_sift(img1: np.ndarray, img2: np.ndarray):
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(k1) < 3 or len(k2) < 3:
        return [], [], []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    return pts1, pts2, good


def estimate_affine_ransac(pts1: np.ndarray, pts2: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if pts1 is None or pts2 is None or len(pts1) < 3 or len(pts2) < 3:
        return None, None
    M, inliers = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
    return M, inliers


def warp_affine(img: np.ndarray, M: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(img, M, (out_shape[1], out_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def align_with_features(fixed: torch.Tensor, moving: torch.Tensor, extractor: Optional[FeatureExtractor] = None):
    """
    fixed, moving: [1,1,H,W] torch float tensors in [0,1]
    extractor: FeatureExtractor with trained SSL U-Net. If None, use raw images.
    Returns: aligned_moving_img (np.uint8), affine matrix M (2x3), inlier mask
    """
    if extractor is not None:
        with torch.no_grad():
            f_feat = extractor.extract(fixed)
            m_feat = extractor.extract(moving)
        img1 = features_to_keyimage(f_feat)
        img2 = features_to_keyimage(m_feat)
    else:
        img1 = _to_numpy_img(fixed)
        img2 = _to_numpy_img(moving)

    pts1, pts2, matches = detect_and_match_sift(img1, img2)
    M, inliers = estimate_affine_ransac(pts1, pts2)
    if M is None:
        # Fallback: identity
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        inliers = None
    aligned = warp_affine(_to_numpy_img(moving), M, _to_numpy_img(fixed).shape)
    return aligned, M, inliers
