import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_data(data_dir: Path) -> None:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_csv = raw_dir / "train.csv"
    test_csv = raw_dir / "test.csv"

    if train_csv.exists() and test_csv.exists():
        logger.info("Data files already exist, skipping download")
        return

    logger.info("Attempting to pull data from DVC...")
    try:
        subprocess.run(
            ["dvc", "pull"],
            cwd=data_dir.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Successfully pulled data from DVC")
        if train_csv.exists() and test_csv.exists():
            return
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"DVC pull failed: {e}, falling back to Kaggle API")

    logger.info("Falling back to Kaggle API...")
    from sentiment_span_extractor.data.download import download_kaggle_competition

    download_kaggle_competition(raw_dir)
