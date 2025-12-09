import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from config import Config
from data import HemibrainDataLoader
from simulation import MisalignmentSimulator, FIBSEMArtifactSimulator
from models import ArtifactRemovalUNet, ImprovedVoxelMorph
from baselines import TraditionalMethods
from metrics import Metrics
from losses import PerceptualLoss, gradient_loss
from tracking import MLflowTracker


class DeepAlignPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger("DeepAlign.Pipeline")
        self.device = torch.device(config.device)
        self.logger.info(f"✓ Device: {self.device}")

        self.data_loader = HemibrainDataLoader(config.mip_level)
        self.artifact_sim = FIBSEMArtifactSimulator(config.curtain_intensity, config.charging_bias)
        self.misalign_sim = MisalignmentSimulator(config.max_translation, config.max_rotation)
        self.traditional = TraditionalMethods()
        self.metrics = Metrics()
        self.mlflow = MLflowTracker(config.experiment_name, config.run_name)

        self.ssl_model: Optional[ArtifactRemovalUNet] = None
        self.reg_model: Optional[ImprovedVoxelMorph] = None
        self.feature_loss: Optional[PerceptualLoss] = None

    def train_ssl(self):
        self.logger.info("=" * 70)
        self.logger.info("STAGE 1: Artifact Removal Training")
        self.logger.info("=" * 70)

        self.ssl_model = ArtifactRemovalUNet(self.config.ssl_features).to(self.device)
        optimizer = torch.optim.Adam(self.ssl_model.parameters(), lr=self.config.learning_rate_ssl)
        criterion = nn.MSELoss()
        self.ssl_model.train()

        for epoch in range(self.config.num_epochs_ssl):
            epoch_loss = 0.0
            n_batches = max(1, self.config.num_training_samples // self.config.num_epochs_ssl)
            for _ in range(n_batches):
                clean = self.data_loader.get_chunk(self.config.hemibrain_roi, self.config.vol_shape)
                corrupted = self.artifact_sim.apply(clean)
                t_clean = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0).to(self.device)
                t_corrupt = torch.from_numpy(corrupted).unsqueeze(0).unsqueeze(0).to(self.device)

                optimizer.zero_grad()
                restored = self.ssl_model(t_corrupt)
                loss = criterion(restored, t_clean)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            self.mlflow.log_metric("ssl_train_loss", avg_loss, step=epoch)
            if epoch % 5 == 0:
                self.logger.info(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

        self.logger.info("✓ SSL training complete")
        self.ssl_model.encoder.eval()
        for param in self.ssl_model.encoder.parameters():
            param.requires_grad = False
        self.mlflow.log_model(self.ssl_model, "ssl_model")

    def train_registration(self):
        self.logger.info("=" * 70)
        self.logger.info("STAGE 3: Registration Training")
        self.logger.info("=" * 70)

        if self.ssl_model is None:
            raise RuntimeError("Train SSL first!")

        self.reg_model = ImprovedVoxelMorph(self.config.vol_shape, self.config.reg_features).to(self.device)
        self.feature_loss = PerceptualLoss(self.ssl_model.encoder)
        optimizer = torch.optim.Adam(self.reg_model.parameters(), lr=self.config.learning_rate_reg)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        self.reg_model.train()
        best_loss = float('inf')

        for epoch in range(self.config.num_epochs_registration):
            epoch_loss = 0.0
            epoch_improve = 0.0
            n_batches = max(1, self.config.num_training_samples // self.config.num_epochs_registration)

            for _ in range(n_batches):
                fixed = self.data_loader.get_chunk(self.config.hemibrain_roi, self.config.vol_shape)
                moving, _ = self.misalign_sim.apply_misalignment(fixed)
                moving_corrupt = self.artifact_sim.apply(moving)

                t_fixed = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).to(self.device)
                t_moving = torch.from_numpy(moving_corrupt).unsqueeze(0).unsqueeze(0).to(self.device)

                optimizer.zero_grad()
                moved, flow = self.reg_model(t_moving, t_fixed)

                loss_int = F.mse_loss(moved, t_fixed)
                loss_feat = self.feature_loss(moved, t_fixed)
                loss_smooth = gradient_loss(flow)

                total_loss = (self.config.lambda_intensity * loss_int +
                              self.config.lambda_feature * loss_feat +
                              self.config.lambda_smooth * loss_smooth)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reg_model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += total_loss.item()

                with torch.no_grad():
                    mse_before = F.mse_loss(t_moving, t_fixed).item()
                    mse_after = F.mse_loss(moved, t_fixed).item()
                    improve = (mse_before - mse_after) / (mse_before + 1e-8) * 100
                    epoch_improve += improve

            avg_loss = epoch_loss / n_batches
            avg_improve = epoch_improve / n_batches
            scheduler.step(avg_loss)

            self.mlflow.log_metrics({
                "reg_train_loss": avg_loss,
                "reg_improvement_pct": avg_improve,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)

            if avg_loss < best_loss:
                best_loss = avg_loss
            if epoch % 5 == 0:
                self.logger.info(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Improve: {avg_improve:+.2f}%")

        self.logger.info("✓ Registration training complete")
        self.mlflow.log_model(self.reg_model, "registration_model")

    def evaluate(self):
        self.logger.info("=" * 70)
        self.logger.info("EVALUATION")
        self.logger.info("=" * 70)

        if self.reg_model is None:
            raise RuntimeError("Train registration first!")

        self.reg_model.eval()
        all_metrics = {m: {'mse': [], 'mae': [], 'ncc': [], 'psnr': []}
                       for m in ['before', 'deepalign', 'correlation', 'mse_search']}

        with torch.no_grad():
            for i in range(self.config.num_test_pairs):
                fixed = self.data_loader.get_chunk(self.config.hemibrain_roi, self.config.vol_shape)
                moving, _ = self.misalign_sim.apply_misalignment(fixed, translation=(7, 8, -6), rotation_angle=1.5)
                moving_corrupt = self.artifact_sim.apply(moving)

                t_fixed = torch.from_numpy(fixed).unsqueeze(0).unsqueeze(0).to(self.device)
                t_moving = torch.from_numpy(moving_corrupt).unsqueeze(0).unsqueeze(0).to(self.device)
                moved_deep, _ = self.reg_model(t_moving, t_fixed)
                aligned_deep = moved_deep.cpu().numpy().squeeze()

                aligned_corr = self.traditional.correlation_search(fixed, moving_corrupt)
                aligned_mse = self.traditional.mse_search(fixed, moving_corrupt)

                results = {'before': moving_corrupt, 'deepalign': aligned_deep,
                           'correlation': aligned_corr, 'mse_search': aligned_mse}

                for method, aligned in results.items():
                    all_metrics[method]['mse'].append(self.metrics.mse(fixed, aligned))
                    all_metrics[method]['mae'].append(self.metrics.mae(fixed, aligned))
                    all_metrics[method]['ncc'].append(self.metrics.ncc(fixed, aligned))
                    all_metrics[method]['psnr'].append(self.metrics.psnr(fixed, aligned))

                if i == 0:
                    avg_metrics = {m: {k: v[0] for k, v in all_metrics[m].items()} for m in all_metrics.keys()}
                    fig = self.mlflow.create_comparison_figure(fixed, moving_corrupt, results, avg_metrics)
                    self.mlflow.log_figure(fig, "alignment_comparison.png")

        agg = {method: {met: np.mean(vals) for met, vals in metrics.items()}
               for method, metrics in all_metrics.items()}

        for method, metrics in agg.items():
            for metric_name, value in metrics.items():
                self.mlflow.log_metric(f"{method}_{metric_name}_mean", value)

        fig = self.mlflow.create_metrics_plot(agg)
        self.mlflow.log_figure(fig, "metrics_comparison.png")
        return agg