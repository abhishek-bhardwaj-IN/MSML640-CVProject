import logging
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, shift as nd_shift, rotate
from typing import Tuple, Dict, Optional

class MisalignmentSimulator:
    def __init__(self, max_translation: int = 10, max_rotation: float = 2.0):
        self.max_translation = max_translation
        self.max_rotation = max_rotation

    def apply_misalignment(self, volume: np.ndarray,
                          translation: Optional[Tuple[int, int, int]] = None,
                          rotation_angle: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
        d, h, w = volume.shape

        if translation is None:
            translation = tuple(np.random.randint(-self.max_translation,
                                                 self.max_translation + 1, size=3))

        misaligned = nd_shift(volume, translation, order=1, mode='constant', cval=0)

        if rotation_angle is None:
            rotation_angle = np.random.uniform(-self.max_rotation, self.max_rotation)

        if abs(rotation_angle) > 0.1:
            for z in range(d):
                misaligned[z] = rotate(misaligned[z], rotation_angle,
                                      order=1, mode='constant', cval=0, reshape=False)

        params = {'translation': translation, 'rotation_deg': float(rotation_angle)}
        return misaligned, params

class FIBSEMArtifactSimulator:
    def __init__(self, curtain_intensity: float = 0.35, charging_bias: float = 0.25):
        self.curtain_intensity = curtain_intensity
        self.charging_bias = charging_bias

    def add_curtaining(self, volume: np.ndarray) -> np.ndarray:
        d, h, w = volume.shape
        num_curtains = np.random.randint(10, 25)
        positions = np.random.choice(w, num_curtains, replace=False)
        mask = np.ones(w, dtype=np.float32)

        for pos in positions:
            width = np.random.randint(2, 6)
            intensity = np.random.uniform(0.2, 0.6)
            start = max(0, pos - width // 2)
            end = min(w, pos + width // 2)
            mask[start:end] *= intensity

        mask = gaussian_filter(mask, sigma=2.5)
        mask_3d = np.tile(mask[np.newaxis, np.newaxis, :], (d, h, 1))
        return np.clip(volume * mask_3d, 0, 1).astype(np.float32)

    def add_charging(self, volume: np.ndarray) -> np.ndarray:
        d, h, w = volume.shape
        low_res = (max(2, d // 8), max(2, h // 8), max(2, w // 8))
        grid = np.random.uniform(-self.charging_bias, self.charging_bias, size=low_res)
        zoom_factors = (d / low_res[0], h / low_res[1], w / low_res[2])
        field = zoom(grid, zoom_factors, order=3)[:d, :h, :w]
        return np.clip(volume + field, 0, 1).astype(np.float32)

    def apply(self, volume: np.ndarray) -> np.ndarray:
        v = self.add_curtaining(volume)
        v = self.add_charging(v)
        return v
