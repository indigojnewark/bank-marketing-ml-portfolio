from __future__ import annotations

import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a small number of engineered features with clear rationale."""
    df = df.copy()

    # In the UCI Bank Marketing dataset, pdays == 999 means "not previously contacted".
    if "pdays" in df.columns:
        df["pdays_clean"] = df["pdays"].where(df["pdays"] != 999, other=pd.NA)
        df["previously_contacted"] = (df["pdays"] != 999).astype(int)

    if "campaign" in df.columns:
        df["log_campaign"] = (df["campaign"] + 1).astype(float).transform("log")

    if "previous" in df.columns:
        df["log_previous"] = (df["previous"] + 1).astype(float).transform("log")

    if "campaign" in df.columns and "previous" in df.columns:
        df["contacts_since_previous_ratio"] = df["campaign"] / (df["previous"] + 1)

    return df
