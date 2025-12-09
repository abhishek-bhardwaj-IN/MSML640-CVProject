import logging
import numpy as np
from scipy.ndimage import shift as nd_shift

class TraditionalMethods:
    def __init__(self):
        self.logger = logging.getLogger("DeepAlign.Traditional")

    def correlation_search(self, fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
        best_shift = (0, 0, 0)
        best_corr = -float('inf')
        search_range = 12
        for dz in range(-search_range, search_range + 1, 2):
            for dy in range(-search_range, search_range + 1, 2):
                for dx in range(-search_range, search_range + 1, 2):
                    shifted = nd_shift(moving, (dz, dy, dx), order=1, mode='constant', cval=0)
                    corr = np.sum(fixed * shifted)
                    if corr > best_corr:
                        best_corr = corr
                        best_shift = (dz, dy, dx)
        return nd_shift(moving, best_shift, order=1, mode='constant', cval=0)

    def mse_search(self, fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
        best_shift = (0, 0, 0)
        best_mse = float('inf')
        search_range = 12
        for dz in range(-search_range, search_range + 1, 2):
            for dy in range(-search_range, search_range + 1, 2):
                for dx in range(-search_range, search_range + 1, 2):
                    shifted = nd_shift(moving, (dz, dy, dx), order=1, mode='constant', cval=0)
                    mse = np.mean((fixed - shifted) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_shift = (dz, dy, dx)
        return nd_shift(moving, best_shift, order=1, mode='constant', cval=0)