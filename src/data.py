import logging
import numpy as np
from typing import Tuple
from scipy.ndimage import gaussian_filter

try:
    from cloudvolume import CloudVolume

    CLOUDVOLUME_AVAILABLE = True
except ImportError:
    CLOUDVOLUME_AVAILABLE = False


class HemibrainDataLoader:
    def __init__(self, mip: int = 2):
        self.logger = logging.getLogger("DeepAlign.Data")
        self.mip = mip

        if not CLOUDVOLUME_AVAILABLE:
            self.logger.warning("CloudVolume unavailable - using synthetic data")
            self.vol_em = None
            return

        try:
            self.em_path = 'gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg'
            self.logger.info(f"Connecting to Hemibrain...")
            self.vol_em = CloudVolume(
                self.em_path, mip=self.mip, use_https=True,
                fill_missing=True, parallel=True, progress=False
            )
            self.logger.info(f"✓ Connected! Shape: {self.vol_em.shape}")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.vol_em = None

    def get_chunk(self, center: Tuple[int, int, int], size: Tuple[int, int, int]) -> np.ndarray:
        if self.vol_em is None:
            return self._generate_synthetic(size)

        d, h, w = size
        cx, cy, cz = center
        x_min, x_max = int(cx - w // 2), int(cx + w // 2)
        y_min, y_max = int(cy - h // 2), int(cy + h // 2)
        z_min, z_max = int(cz - d // 2), int(cz + d // 2)

        try:
            cutout = self.vol_em[x_min:x_max, y_min:y_max, z_min:z_max]
            data = np.transpose(np.squeeze(np.array(cutout)), (2, 1, 0))
            data = data.astype(np.float32) / 255.0
            return data
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            return self._generate_synthetic(size)

    def _generate_synthetic(self, size: Tuple[int, int, int]) -> np.ndarray:
        d, h, w = size
        volume = np.random.rand(*size).astype(np.float32) * 0.2
        for sigma in [4, 8, 16]:
            blobs = gaussian_filter(np.random.rand(*size), sigma=sigma)
            volume += blobs * 0.3

        num_membranes = 8
        for _ in range(num_membranes):
            y = np.random.randint(h // 4, 3 * h // 4)
            x = np.random.randint(w // 4, 3 * w // 4)
            thickness = 3
            volume[:, y - thickness:y + thickness, x - thickness:x + thickness] += 0.6

        volume = np.clip(volume, 0, 1)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        return volume