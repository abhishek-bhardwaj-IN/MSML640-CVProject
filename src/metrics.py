import numpy as np

class Metrics:
    @staticmethod
    def mse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean((a - b) ** 2))

    @staticmethod
    def mae(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    @staticmethod
    def ncc(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = (a - np.mean(a)) / (np.std(a) + 1e-8)
        b_norm = (b - np.mean(b)) / (np.std(b) + 1e-8)
        return float(np.mean(a_norm * b_norm))

    @staticmethod
    def psnr(a: np.ndarray, b: np.ndarray) -> float:
        mse = np.mean((a - b) ** 2)
        if mse < 1e-10:
            return 100.0
        return float(20 * np.log10(1.0 / np.sqrt(mse)))
    