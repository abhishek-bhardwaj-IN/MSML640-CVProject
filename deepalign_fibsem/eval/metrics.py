import numpy as np
import torch
import torch.nn.functional as F


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2 * inter) / (pred.sum() + gt.sum() + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


def jacobian_determinant_2d(flow: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian determinant of 2D displacement field.
    flow: [B,2,H,W] with pixel displacements (dx,dy)
    returns: [B,H,W]
    """
    dx = flow[:, 0:1]
    dy = flow[:, 1:1+1]
    dxx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    dxy = dx[:, :, 1:, :] - dx[:, :, :-1, :]
    dyx = dy[:, :, :, 1:] - dy[:, :, :, :-1]
    dyy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    # pad to original size
    dxx = F.pad(dxx, (0, 1, 0, 0))
    dxy = F.pad(dxy, (0, 0, 0, 1))
    dyx = F.pad(dyx, (0, 1, 0, 0))
    dyy = F.pad(dyy, (0, 0, 0, 1))
    # Jacobian of transform x' = x + dx, y' = y + dy is:
    # J = [[1 + d(dx)/dx, d(dx)/dy], [d(dy)/dx, 1 + d(dy)/dy]]
    J11 = 1 + dxx
    J12 = dxy
    J21 = dyx
    J22 = 1 + dyy
    detJ = J11 * J22 - J12 * J21
    return detJ.squeeze(1)


def percent_nonpositive_jacobian(detJ: torch.Tensor) -> float:
    nonpos = (detJ <= 0).float().mean().item() * 100.0
    return nonpos
