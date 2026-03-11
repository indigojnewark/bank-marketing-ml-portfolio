from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml

from .utils import ensure_dir, get_project_root

logger = logging.getLogger(__name__)


DATASET_NAME = "bank-marketing"
DATASET_VERSION = 1
TARGET_COLUMN_OPTIONS = ("y", "class")


def download_dataset(raw_dir: Path | None = None) -> Path:
    """Download dataset via OpenML and save to data/raw."""
    if raw_dir is None:
        raw_dir = get_project_root() / "data" / "raw"
    ensure_dir(raw_dir)

    out = raw_dir / f"{DATASET_NAME}-v{DATASET_VERSION}.parquet"
    if out.exists():
        logger.info("Dataset already at %s", out)
        return out

    ds = fetch_openml(DATASET_NAME, version=DATASET_VERSION, as_frame=True)
    df = ds.frame.copy()
    df.to_parquet(out, index=False)
    logger.info("Saved raw dataset to %s [rows=%d]", out, len(df))
    return out


def load_raw_data(raw_file: Path) -> pd.DataFrame:
    df = pd.read_parquet(raw_file)
    logger.info("Loaded raw dataset from %s [rows=%d]", raw_file, len(df))
    return df


def identify_target_column(df: pd.DataFrame) -> str:
    for candidate in TARGET_COLUMN_OPTIONS:
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find target column in df columns")


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    target_col = identify_target_column(df)

    X = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()

    y = y_raw.replace({"yes": 1, "no": 0, True: 1, False: 0}).astype("int64")
    return X, y, target_col
