import joblib
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from .data_loader import split_features_target
from .features import add_domain_features
from .models import get_model_specs
from .preprocessing import PreprocessConfig, build_preprocessor
from .evaluation import compute_classification_metrics, save_metrics
from .utils import DEFAULT_RANDOM_STATE, ensure_dir


def train_validate_test_pipeline(df, results_dir=None, random_state=DEFAULT_RANDOM_STATE, n_iter=10):
    if results_dir is None:
        results_dir = Path("results")
    else:
        results_dir = Path(results_dir)

    ensure_dir(results_dir)
    model_dir = results_dir / "models"
    metrics_dir = results_dir / "metrics"
    ensure_dir(model_dir)
    ensure_dir(metrics_dir)

    X_raw, y, _ = split_features_target(df)
    X_feat = add_domain_features(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, stratify=y, random_state=random_state
    )

    config = PreprocessConfig()
    preprocessor = build_preprocessor(X_train, config)

    best_auc = -1.0
    best_name = ""
    best_pipeline = None

    for name, spec in get_model_specs().items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("classifier", spec.estimator)])

        if spec.param_distributions:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=spec.param_distributions,
                n_iter=n_iter,
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
                random_state=random_state,
                refit=True,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
        else:
            model = pipeline.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_pipeline = model

    if best_pipeline is None:
        raise RuntimeError("No model trained")

    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test.to_numpy(), y_prob)

    model_path = model_dir / f"{best_name}.joblib"
    joblib.dump(best_pipeline, model_path)

    save_metrics(metrics, metrics_dir / f"{best_name}_test_metrics.json")
    save_metrics(
        {"best_model": best_name, "test_roc_auc": metrics.get("roc_auc", 0.0)},
        metrics_dir / "final_model_selection.json",
    )

    return metrics, {"best_model": best_name, "test_roc_auc": best_auc}, best_name, model_path
