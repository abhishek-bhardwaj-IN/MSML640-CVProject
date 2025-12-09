import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

class MLflowTracker:
    def __init__(self, experiment_name: str, run_name: str):
        self.logger = logging.getLogger("DeepAlign.MLflow")
        self.active = MLFLOW_AVAILABLE

        if self.active:
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow experiment: {experiment_name}")

    def log_params(self, params: Dict):
        if self.active:
            for k, v in params.items():
                mlflow.log_param(k, v)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        if self.active:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.active:
            mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: nn.Module, path: str):
        if self.active:
            mlflow.pytorch.log_model(model, path)

    def log_figure(self, fig: plt.Figure, filename: str):
        if not self.active:
            plt.close(fig)
            return

        fig_path = f"figures/{filename}"
        Path("figures").mkdir(exist_ok=True)
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(fig_path)
        plt.close(fig)

    def create_comparison_figure(self, fixed: np.ndarray, moving: np.ndarray,
                                 results: Dict[str, np.ndarray],
                                 metrics: Dict[str, Dict[str, float]]) -> plt.Figure:
        mid_z = fixed.shape[0] // 2
        methods = ['before'] + [k for k in results.keys() if k != 'before']
        n = len(methods)

        fig, axes = plt.subplots(3, n, figsize=(5*n, 15))

        for idx, method in enumerate(methods):
            aligned = results[method]
            m = metrics[method]

            # Row 1: Aligned image
            axes[0, idx].imshow(aligned[mid_z], cmap='gray', vmin=0, vmax=1)
            title = f'{method.upper()}\nMSE: {m["mse"]:.5f} | NCC: {m["ncc"]:.3f}'
            axes[0, idx].set_title(title, fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')

            # Row 2: Error map
            error = np.abs(fixed[mid_z] - aligned[mid_z])
            axes[1, idx].imshow(error, cmap='hot', vmin=0, vmax=0.4)
            axes[1, idx].set_title(f'Error Map\nMAE: {m["mae"]:.5f}', fontsize=11)
            axes[1, idx].axis('off')

            # Row 3: Checkerboard comparison
            checker = self._create_checkerboard(fixed[mid_z], aligned[mid_z])
            axes[2, idx].imshow(checker, cmap='gray', vmin=0, vmax=1)
            axes[2, idx].set_title(f'Checkerboard\nPSNR: {m["psnr"]:.2f} dB', fontsize=11)
            axes[2, idx].axis('off')

        plt.tight_layout()
        return fig

    def _create_checkerboard(self, img1: np.ndarray, img2: np.ndarray, block_size: int = 16) -> np.ndarray:
        h, w = img1.shape
        checker = np.copy(img1)
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 1:
                    checker[i:i+block_size, j:j+block_size] = img2[i:i+block_size, j:j+block_size]
        return checker

    def create_metrics_plot(self, metrics: Dict[str, Dict[str, float]]) -> plt.Figure:
        methods = list(metrics.keys())
        metric_names = ['mse', 'mae', 'ncc', 'psnr']
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

        for idx, metric in enumerate(metric_names):
            values = [metrics[m].get(metric, 0) for m in methods]
            bars = axes[idx].bar(methods, values, color=colors)
            axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel(metric.upper(), fontsize=12)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(axis='y', alpha=0.3)
            for bar in bars:
                h = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        return fig
    