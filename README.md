# Bank Marketing ML Portfolio Project

A polished, reproducible machine learning portfolio repository based on the **UCI Bank Marketing** dataset (via OpenML). The goal is to build a credible end-to-end student project that demonstrates strong ML workflow skills, clean code, and thoughtful decision-making (including handling dataset leakage).

## Problem

**Task:** Predict whether a customer will subscribe to a term deposit after a marketing campaign call.

**Type:** Binary classification, structured tabular data, class imbalance.

**Why this matters:** It’s a genuinely realistic ML scenario: messy real-world features, categorical variables, mixed scales, imbalance, and the temptation to use a highly predictive but **leaky** feature (`duration`) that should not be used for prospecting.

## Dataset

- **Source:** UCI Bank Marketing (downloaded via scikit-learn OpenML fetcher)
- **Size:** ~41k rows, dozens of features
- **Target column:** `y` / `class` (converted to 0/1)
- **Leakage note:** `duration` is known to be strongly predictive but is not available at decision time; this repo **drops** it from the feature set in the preprocessing pipeline.

## Methods

This project is structured as a lightweight Python package (`src/`) with scripts and notebooks layered on top.

### Workflow demonstrated
1. Data ingestion (OpenML download + saved raw file)
2. Exploratory analysis (EDA notebook)
3. Feature engineering (domain-inspired features only, no leakage)
4. Train / validation / test split (stratified)
5. Multiple model families:
   - Dummy baseline (most frequent)
   - Logistic Regression (balanced class weights)
   - Random Forest (balanced class weights)
   - XGBoost (optional, if installed)
6. Hyperparameter tuning via `RandomizedSearchCV` (ROC AUC scoring)
7. Model comparison on validation set
8. Final model selection and retraining on train+val
9. Test evaluation
10. Visualisation + interpretability:
    - ROC & PR curves
    - Confusion matrix
    - Feature importance (for supported models)

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_model_experiments.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── utils.py
│   └── __init__.py
├── scripts/
│   ├── download_data.py
│   ├── train_model.py
│   └── evaluate_model.py
└── results/
    ├── figures/
    ├── metrics/
    └── models/
```

## Results (example)

Because model performance depends on package versions, hardware, and random seed, your exact numbers may differ. On a typical run, the best model is usually a tree ensemble (Random Forest / XGBoost) with **ROC AUC around 0.86–0.89** on the held-out test set.

Artifacts saved to `results/` include:
- `metrics/final_model_selection.json` (selected model + validation AUC)
- `metrics/<model>_test_metrics.json` (test metrics)
- `figures/<model>_roc_pr_test.png`
- `figures/<model>_cm_test.png`
- `figures/<model>_feature_importance.png` (when supported)

## How to run

### 1) Install dependencies

Using a virtual environment is recommended.

```bash
pip install -r requirements.txt
```

### 2) Download data

```bash
python scripts/download_data.py
```

### 3) Train models and produce reports

```bash
python scripts/train_model.py
```

### 4) Evaluate a saved model (optional)

```bash
python scripts/evaluate_model.py --model_path results/models/<best_model>.joblib
```

## What this repo demonstrates

- Clean, modular Python with docstrings and type hints
- Serious project structure (not a single notebook)
- Proper ML hygiene: leakage awareness, imbalance handling, validation selection
- Reproducible experiments: fixed seed, saved artifacts, script-driven workflow
- Strong visualisation and reporting practices

## Tech stack

- Python
- pandas / numpy
- scikit-learn
- matplotlib / seaborn
- XGBoost (optional)
