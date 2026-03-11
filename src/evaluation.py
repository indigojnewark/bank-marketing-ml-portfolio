import json
from pathlib import Path

import numpy as np


def compute_classification_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_metrics(metrics, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))


def plot_roc_pr_curves(y_true, y_prob, title, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("[plot placeholder]")


def plot_confusion_matrix(y_true, y_prob, title, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("[plot placeholder]")


def plot_feature_importance(estimator, feature_names, title, output_path, top_k=20):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("[plot placeholder]")
