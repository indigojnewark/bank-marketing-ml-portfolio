from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for preprocessing.

    Notes:
    - 'duration' is not used as a feature because it is a known leakage channel.
    """

    drop_columns: Sequence[str] = ("duration",)


def build_preprocessor(X: pd.DataFrame, config: PreprocessConfig | None = None) -> ColumnTransformer:
    """Build a ColumnTransformer that scales numeric features and one-hot encodes categorical features."""
    if config is None:
        config = PreprocessConfig()

    X = X.copy()
    X = X.drop(columns=[c for c in config.drop_columns if c in X.columns], errors="ignore")

    numeric_features = [col for col, dtype in X.dtypes.items() if dtype.kind in "iu"]
    numeric_features += [col for col, dtype in X.dtypes.items() if dtype.kind == "f"]

    categorical_features = [col for col, dtype in X.dtypes.items() if dtype == "object"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, sorted(set(numeric_features))),
            ("cat", categorical_transformer, sorted(set(categorical_features))),
        ]
    )

    return preprocessor
