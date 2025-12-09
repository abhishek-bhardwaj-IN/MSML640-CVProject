
# DeepAlign-FIBSEM (Hemibrain) – UV-based Pipeline Runner

## Overview

This project trains and evaluates a three-stage alignment pipeline for FIB-SEM data using synthetic artifacts and misalignments.

**Entry point:** `src/run.py` orchestrates training (SSL features + deformable registration) and evaluation, and logs results to MLflow and figures to disk.

## Key Features

- Synthetic data fallback when Hemibrain CloudVolume is unavailable (runs anywhere)
- Self-supervised artifact removal network to learn robust features
- Deformable registration guided by learned features, with classic baselines for comparison
- Tracking with MLflow (parameters, metrics, models, artifacts)

## Environment (UV-based)

- Requires Python 3.10+
- Recommended: [Astral's uv](https://github.com/astral-sh/uv) for environment and package management

### Install uv

| Platform | Command |
|----------|---------|
| macOS (Homebrew) | `brew install uv` |
| Universal (shell) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

Verify installation:
```bash 
  uv --version
````


### Create Environment and Install Dependencies

1. From the project root, create a virtual environment:
   ```bash
   uv venv .venv
   ```

2. Activate it:
   - **macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

3. Install packages from requirements.txt:
   ```bash
   uv pip install -r requirements.txt
   ```

## How to Run

The pipeline is designed to be run from the `src` directory (so `logs/` and `figures/` are created in `src/`).

### Option A (Recommended)

Change into `src` and run:

```bash
cd src 
uv run python run.py
```


### Option B (From Project Root)

```bash
uv run python -m src.run
```


> **Note:** When run from the project root, `logs/` and `figures/` will be created at the project root instead of under `src/`.

## Results and Outputs

| Output | Location |
|--------|----------|
| MLflow tracking directory | `src/mlruns/` (created automatically when run from `src`) |
| Logs | `src/logs/deepalign_YYYYMMDD_HHMMSS.log` |
| Alignment comparison figure | `src/figures/alignment_comparison.png` |
| Metrics comparison figure | `src/figures/metrics_comparison.png` |

## Viewing the MLflow UI

From the same working directory used to run the pipeline (e.g., `src/`):
```bash
uv run mlflow ui
```
```txt
src/
├── run.py              # Entry point
├── pipeline.py         # Training/evaluation orchestration
├── config.py           # Configuration dataclass
├── data.py             # Hemibrain loader + synthetic fallback
├── models.py           # Networks (artifact removal UNet, ImprovedVoxelMorph)
├── baselines.py        # Traditional baselines
├── losses.py           # Losses, smoothness
├── metrics.py          # Evaluation metrics
├── tracking.py         # MLflow logging and figure generation
├── figures/            # Generated figures
├── logs/               # Run logs
└── mlruns/             # MLflow runs
```