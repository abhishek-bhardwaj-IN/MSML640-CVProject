DeepAlign-FIBSEM: Self-Supervised, Artifact-Robust Alignment for FIB-SEM

Overview
- Goal: Robust alignment of large FIB-SEM volumes using a three-stage pipeline.
- Stages:
  1) Self-Supervised Artifact-Invariant Feature Learning (U-Net denoising on synthetic artifacts).
  2) Coarse Global Affine Alignment using SIFT + RANSAC on learned feature maps.
  3) Fine Deformable Registration guided by the learned features (feature-space loss).
- Baselines: (a) SIFT+RANSAC on raw images, (b) VoxelMorph-like deformable with NCC on raw.
- Metrics: Dice, IoU, Jacobian determinant non-positive percentage.

Environment
- Python >= 3.10 recommended.
- Windows, Linux, macOS supported. GPU optional but recommended.

Install
1) Create and activate a virtual environment.
2) Install requirements:
   pip install -r requirements.txt

Project Structure
- deepalign_fibsem/
  - data/dataset.py: SSL dataset with synthetic artifact generator; pair dataset utilities.
  - models/unet.py: U-Net for artifact removal (encoder used as feature extractor).
  - models/registration_net.py: Lightweight flow-predictor and warp utilities.
  - features/extractor.py: Loads SSL model and extracts encoder features.
  - alignment/affine.py: SIFT keypoints + RANSAC affine using feature maps.
  - alignment/deformable.py: Training loop for deformable reg with feature-space loss.
  - baselines/: SIFT+RANSAC on raw; VoxelMorph-like NCC training.
  - eval/metrics.py: Dice, IoU, Jacobian determinant utilities.
- scripts/
  - train_ssl.py: Train SSL U-Net on synthetic artifacts.
  - train_deformable.py: Train deformable net with feature-space loss.
  - make_benchmark.py: Build a toy benchmark by synthesizing misaligned pairs.
- deepalign_fibsem/pipeline/run_pipeline.py: Orchestrates the 3-stage pipeline.

Data
- You can start with any EM 2D slices (PNG/TIF). The SSL stage synthesizes artifacts, so only clean-looking EM images are needed.
- For evaluation, use scripts/make_benchmark.py to create fixed/moving pairs from clean inputs.
- For real datasets (e.g., Janelia OpenOrganelle / FlyEM), adapt loaders to your data layout. This repo demonstrates the core methods on 2D slices; extend to 3D as needed.

Usage
1) Train SSL U-Net (Stage 1)
   python scripts/train_ssl.py <clean_images_dir> --out checkpoints/ssl_unet.pt --epochs 20 --batch 8

2) Coarse Affine Alignment (Stage 2) using learned features
   python -m deepalign_fibsem.pipeline.run_pipeline <fixed.png> <moving.png> --ssl_ckpt checkpoints/ssl_unet.pt --out aligned_affine.png

3) Train Deformable Registration (Stage 3)
   # First, create a toy benchmark from clean images
   python scripts/make_benchmark.py <clean_images_dir> bench/
   # Train deformable reg guided by SSL features
   python scripts/train_deformable.py bench/fixed/*.png bench/moving/*.png --ssl_ckpt checkpoints/ssl_unet.pt --out checkpoints/deformable.pt

4) Full Pipeline (Affine + Deformable)
   python -m deepalign_fibsem.pipeline.run_pipeline <fixed.png> <moving.png> --ssl_ckpt checkpoints/ssl_unet.pt --deform_ckpt checkpoints/deformable.pt --out aligned.png

Baselines
- SIFT + RANSAC on raw images:
  python -m deepalign_fibsem.baselines.sift_ransac_baseline <fixed.png> <moving.png> --out aligned_raw.png
- VoxelMorph-like with NCC on raw images:
  python -m deepalign_fibsem.baselines.voxelmorph_baseline bench/fixed/*.png bench/moving/*.png

Evaluation
- Overlap metrics (Dice, IoU) and deformation plausibility:
  See deepalign_fibsem/eval/metrics.py. For Jacobian, compute percent of non-positive determinant values; target < 0.1%.

Notes
- This reference implementation focuses on 2D slices for clarity and speed. Extending to 3D (3D U-Nets, 3D flows) follows the same design.
- SIFT requires opencv-contrib-python (included in requirements).
- Adjust training hyperparameters to your compute and data.

Citations & Context
- Based on a course project concept: Self-supervised artifact-robust alignment of FIB-SEM volumes. The pipeline mirrors: SSL feature learning -> RANSAC affine -> feature-guided deformable registration.
