import os
import unittest
import numpy as np
import torch

import deepalign_fibsem as pkg
from deepalign_fibsem.models.unet import UNetDenoise
from deepalign_fibsem.models.registration_net import SimpleRegNet, warp
from deepalign_fibsem.features.extractor import FeatureExtractor
from deepalign_fibsem.alignment.affine import align_with_features
from deepalign_fibsem.eval.metrics import dice_iou, jacobian_determinant_2d, percent_nonpositive_jacobian


class TestImportsAndModels(unittest.TestCase):
    def test_package_imports(self):
        self.assertTrue(hasattr(pkg, "__all__"))
        # Import subpackages
        import deepalign_fibsem.data  # noqa: F401
        import deepalign_fibsem.models  # noqa: F401
        import deepalign_fibsem.alignment  # noqa: F401
        import deepalign_fibsem.features  # noqa: F401
        import deepalign_fibsem.eval  # noqa: F401
        import deepalign_fibsem.baselines  # noqa: F401

    def test_unet_forward(self):
        torch.manual_seed(0)
        net = UNetDenoise(in_ch=1, base_ch=16)
        x = torch.rand(1, 1, 64, 64)
        y, feat = net(x)
        self.assertEqual(y.shape, (1, 1, 64, 64))
        # Bottleneck after 3 downsamples (64 -> 8)
        self.assertEqual(feat.shape[-2:], (8, 8))
        self.assertTrue(torch.all((y >= 0) & (y <= 1)))

    def test_regnet_and_warp_identity(self):
        torch.manual_seed(0)
        net = SimpleRegNet(in_ch=2, base_ch=16)
        x = torch.rand(1, 2, 16, 16)
        flow = net(x)
        self.assertEqual(flow.shape, (1, 2, 16, 16))
        # If flow is zero, warp should be identity
        moving = torch.rand(1, 1, 16, 16)
        zflow = torch.zeros_like(flow)
        warped = warp(moving, zflow)
        self.assertTrue(torch.allclose(warped, moving, atol=1e-6))


class TestFeaturesAndAlignment(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_feature_extractor_shapes(self):
        ext = FeatureExtractor(checkpoint_path=None, device="cpu")
        x = torch.rand(1, 1, 64, 64)
        feat = ext.extract(x)
        self.assertEqual(feat.shape[0], 1)
        self.assertEqual(feat.shape[-2:], (8, 8))
        den = ext.denoise(x)
        self.assertEqual(den.shape, (1, 1, 64, 64))

    def test_affine_align_runs(self):
        # Create simple synthetic images rich in keypoints: random dots
        H, W = 128, 128
        fixed = np.zeros((H, W), dtype=np.float32)
        rng = np.random.default_rng(0)
        ys = rng.integers(10, H-10, size=200)
        xs = rng.integers(10, W-10, size=200)
        fixed[ys, xs] = 1.0
        moving = np.zeros_like(fixed)
        dy, dx = 5, -7
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - dy >= 0) & (yy - dy < H) & (xx - dx >= 0) & (xx - dx < W)
        moving[mask] = fixed[yy - dy, xx - dx][mask]
        fixed_t = torch.from_numpy(fixed[None, None, ...])
        moving_t = torch.from_numpy(moving[None, None, ...])
        aligned, M, inliers = align_with_features(fixed_t, moving_t, extractor=None)
        self.assertEqual(aligned.shape, fixed.shape)
        self.assertEqual(M.shape, (2, 3))


class TestMetricsAndTraining(unittest.TestCase):
    def test_metrics_basic(self):
        a = np.zeros((16, 16), dtype=bool)
        b = np.zeros((16, 16), dtype=bool)
        a[:, :8] = True
        b[:, :8] = True
        dice, iou = dice_iou(a, b)
        self.assertAlmostEqual(dice, 1.0, places=6)
        self.assertAlmostEqual(iou, 1.0, places=6)
        # Jacobian for zero flow is 1 everywhere
        flow = torch.zeros(1, 2, 16, 16)
        detJ = jacobian_determinant_2d(flow)
        self.assertTrue(torch.allclose(detJ, torch.ones(1, 16, 16)))
        self.assertEqual(percent_nonpositive_jacobian(detJ), 0.0)

    def test_deformable_train_step(self):
        # Minimal one-epoch training on tiny synthetic pair
        from deepalign_fibsem.alignment.deformable import train_deformable
        # Make a simple image and a slightly shifted version
        H, W = 64, 64
        img = torch.zeros(1, 1, H, W)
        img[:, :, 16:48, 16:48] = 1.0
        img_shift = torch.zeros_like(img)
        img_shift[:, :, 16:48, 18:50] = 1.0
        dataloader = [(img, img_shift)]  # len=1, yields one batch of shape [1,1,H,W]
        extractor = FeatureExtractor(checkpoint_path=None, device="cpu")
        result = train_deformable(dataloader, extractor, device="cpu", epochs=1, lr=1e-3, smooth_weight=0.0)
        self.assertIn("model", result)
        self.assertIn("history", result)
        self.assertTrue(np.isfinite(result["history"]["loss"][0]))


if __name__ == "__main__":
    unittest.main()
