"""Script to download the Bank Marketing dataset."""
import logging

from src.data_loader import download_dataset
from src.utils import configure_logging

def main() -> None:
    configure_logging(logging.INFO)
    download_dataset()

if __name__ == "__main__":
    main()
