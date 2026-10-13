from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = REPO_ROOT / "data"

DATA_FILE_PATH = DATA_PATH / "microgrid_benchmark_7day.csv"

@staticmethod
def get_data_path(make_dir_if_missing=True):
    """
    Returns the absolute path to the benchmark dataset CSV file.
    """
    logger.info(f"Data path set to: {DATA_PATH}")
    if make_dir_if_missing:
        DATA_PATH.mkdir(parents=True, exist_ok=True)
    return DATA_FILE_PATH


