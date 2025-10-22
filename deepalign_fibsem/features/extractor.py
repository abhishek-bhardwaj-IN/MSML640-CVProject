import torch
import torch.nn.functional as F
from ..models.unet import UNetDenoise


class FeatureExtractor:
    """
    Wraps a trained UNetDenoise model and exposes an interface to obtain encoder features.
    """
    def __init__(self, checkpoint_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = UNetDenoise().to(self.device)
        if checkpoint_path is not None and len(str(checkpoint_path)) > 0:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                self.model.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt)
        self.model.eval()

    @torch.no_grad()
    def extract(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        img_tensor: [B,1,H,W] in [0,1]
        returns: bottleneck features [B,C,H/8,W/8]
        """
        img_tensor = img_tensor.to(self.device)
        x1, x2, x3, x4 = self.model.encode(img_tensor)
        return x4

    @torch.no_grad()
    def denoise(self, img_tensor: torch.Tensor) -> torch.Tensor:
        out, _ = self.model(img_tensor.to(self.device))
        return out
