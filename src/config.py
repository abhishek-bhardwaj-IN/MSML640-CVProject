from dataclasses import dataclass
from typing import Tuple, List
from datetime import datetime
import torch

@dataclass
class Config:
    """Central configuration."""
    experiment_name: str = "DeepAlign-FIBSEM-Hemibrain-v2"
    run_name: str = None
    device: str = "auto"

    # Data
    mip_level: int = 2
    vol_shape: Tuple[int, int, int] = (32, 96, 96)
    hemibrain_roi: Tuple[int, int, int] = (4500, 5000, 5000)

    # Training
    batch_size: int = 10
    learning_rate_ssl: float = 2e-4
    learning_rate_reg: float = 1e-4
    num_epochs_ssl: int = 150
    num_epochs_registration: int = 90
    num_training_samples: int = 1000

    # Loss weights
    lambda_smooth: float = 0.5
    lambda_feature: float = 0.3
    lambda_intensity: float = 2.0

    # Artifacts
    curtain_intensity: float = 0.35
    charging_bias: float = 0.25

    # Misalignment
    max_translation: int = 10
    max_rotation: float = 2.0

    # Architecture
    ssl_features: List[int] = None
    reg_features: List[int] = None

    # Eval
    num_test_pairs: int = 5

    def __post_init__(self):
        if self.ssl_features is None:
            self.ssl_features = [32, 64, 128]
        if self.reg_features is None:
            self.reg_features = [32, 64, 128, 256]

        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        if self.run_name is None:
            self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
