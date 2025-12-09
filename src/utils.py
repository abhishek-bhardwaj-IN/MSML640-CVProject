import sys
import logging
from pathlib import Path
from typing import Optional

def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )

    return logging.getLogger("DeepAlign")
