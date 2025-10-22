import argparse
import cv2
import numpy as np
from ..data.dataset import imread_gray
from ..alignment.affine import detect_and_match_sift, estimate_affine_ransac, warp_affine


def run_sift_ransac(fixed_path: str, moving_path: str, out_path: str | None = None):
    fixed = (imread_gray(fixed_path) * 255).astype(np.uint8)
    moving = (imread_gray(moving_path) * 255).astype(np.uint8)
    pts1, pts2, good = detect_and_match_sift(fixed, moving)
    M, inliers = estimate_affine_ransac(pts1, pts2)
    if M is None:
        print("Failed to estimate affine; using identity")
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    aligned = warp_affine(moving, M, fixed.shape)
    if out_path:
        cv2.imwrite(out_path, aligned)
    return aligned, M, inliers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixed")
    parser.add_argument("moving")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_sift_ransac(args.fixed, args.moving, args.out)
