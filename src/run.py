import sys
import logging
from dataclasses import asdict
from datetime import datetime

from config import Config
from utils import setup_logging
from pipeline import DeepAlignPipeline

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/deepalign_{timestamp}.log"
    logger = setup_logging(log_file)

    logger.info("="*70)
    logger.info("DeepAlign-FIBSEM: Hemibrain Training Pipeline")
    logger.info("="*70)

    config = Config()

    logger.info("Configuration:")
    for k, v in asdict(config).items():
        logger.info(f"  {k}: {v}")

    try:
        # Check for MLflow availability wrapper in tracking.py implies soft dependency,
        # but main execution loop relies on it being importable here for context manager
        import mlflow
    except ImportError:
        logger.error("MLflow required. Install: pip install mlflow")
        return 1

    with mlflow.start_run(run_name=config.run_name):
        pipeline = DeepAlignPipeline(config)
        pipeline.mlflow.log_params(asdict(config))

        try:
            pipeline.train_ssl()
            pipeline.train_registration()
            pipeline.evaluate()

            logger.info("="*70)
            logger.info("✓ Pipeline Complete!")
            logger.info("Run 'mlflow ui' to view results at http://localhost:5000")
            logger.info("="*70)
            return 0

        except KeyboardInterrupt:
            logger.warning("Interrupted")
            return 130
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)
            return 1

if __name__ == "__main__":
    sys.exit(main())