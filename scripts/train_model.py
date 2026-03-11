"""Train models and generate metrics."""
import logging
from pathlib import Path

from src.data_loader import download_dataset, load_raw_data
from src.training import train_validate_test_pipeline
from src.utils import configure_logging

def main() -> None:
    configure_logging(logging.INFO)
    df = load_raw_data(download_dataset())
    train_validate_test_pipeline(df, results_dir=Path("results"))

if __name__ == "__main__":
    main()
