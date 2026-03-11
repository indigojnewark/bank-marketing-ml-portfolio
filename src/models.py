from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class ModelSpec:
    estimator: Any
    param_distributions: Dict[str, Any]


def get_model_specs(include_xgboost: bool = True) -> Dict[str, ModelSpec]:
    """Return model specs (estimator + hyperparameter search space)."""
    specs: Dict[str, ModelSpec] = {}

    specs["dummy_majority"] = ModelSpec(
        estimator=DummyClassifier(strategy="most_frequent"),
        param_distributions={},
    )

    specs["logistic_regression"] = ModelSpec(
        estimator=LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced"),
        param_distributions={
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
    )

    specs["random_forest"] = ModelSpec(
        estimator=RandomForestClassifier(n_jobs=-1, class_weight="balanced"),
        param_distributions={
            "classifier__n_estimators": [200, 400, 800],
            "classifier__max_depth": [5, 10, 15, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        },
    )

    if include_xgboost:
        try:
            from xgboost import XGBClassifier  # type: ignore

            specs["xgboost"] = ModelSpec(
                estimator=XGBClassifier(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    n_jobs=-1,
                    tree_method="hist",
                ),
                param_distributions={
                    "classifier__max_depth": [3, 5, 7, 9],
                    "classifier__subsample": [0.6, 0.8, 1.0],
                    "classifier__colsample_bytree": [0.6, 0.8, 1.0],
                    "classifier__learning_rate": [0.01, 0.05, 0.1],
                },
            )
        except Exception:
            pass

    return specs
