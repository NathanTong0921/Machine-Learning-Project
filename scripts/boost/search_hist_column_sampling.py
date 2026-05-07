"""Search histogram bin counts and column sampling for XGBoost and LightGBM.

Declaration:
- This script runs an isolated grid over histogram `max_bin` and per-tree
  feature sampling for the DSAA2011 student outcome three-class task.
- It keeps other model parameters close to the existing XGBoost baseline and
  LightGBM tuned baseline, so the experiment isolates bucket and column-count
  effects instead of becoming a broad hyperparameter search.

Tips:
- Read `f1_macro` first, then inspect `enrolled_f1` because `Enrolled` is the
  minority class.
- Categorical columns are selected from `src.config.CATEGORICAL_COLUMNS`; do not
  split by pandas dtype.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Compatibility shim for older xgboost releases under NumPy 2.x.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns


MAX_BIN_GRID = [16, 32, 64, 128, 255]
COLUMN_FRACTION_GRID = [0.60, 0.75, 0.90, 1.00]

LIGHTGBM_BASE_PARAMS = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "min_child_samples": 40,
    "max_depth": -1,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}

XGBOOST_BASE_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "mlogloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.9,
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return value


def load_data():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train = train_df[TARGET_COL].copy()
    X_test = test_df.drop(columns=[TARGET_COL]).copy()
    y_test = test_df[TARGET_COL].copy()

    numeric_cols, cat_cols = split_feature_columns(X_train.columns)
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    return X_train, y_train, X_test, y_test, numeric_cols, cat_cols


def class_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "dropout_f1": report["Dropout"]["f1-score"],
        "enrolled_precision": report["Enrolled"]["precision"],
        "enrolled_recall": report["Enrolled"]["recall"],
        "enrolled_f1": report["Enrolled"]["f1-score"],
        "graduate_f1": report["Graduate"]["f1-score"],
    }


def make_param_grid(quick):
    max_bins = [16, 255] if quick else MAX_BIN_GRID
    fractions = [0.60, 1.00] if quick else COLUMN_FRACTION_GRID
    return list(ParameterGrid({"max_bin": max_bins, "column_fraction": fractions}))


def make_model(model_name, params, n_jobs):
    model_params = {
        "max_bin": int(params["max_bin"]),
        "colsample_bytree": float(params["column_fraction"]),
        "n_jobs": n_jobs,
    }
    if model_name == "lightgbm":
        return LGBMClassifier(**LIGHTGBM_BASE_PARAMS, **model_params)
    if model_name == "xgboost_hist":
        return XGBClassifier(**XGBOOST_BASE_PARAMS, **model_params)
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_cv(model_name, X_train, y_train_text, cat_cols, cv, param_grid, n_jobs):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_text)
    records = []
    n_features = X_train.shape[1]
    total = len(param_grid)

    for combo_idx, params in enumerate(param_grid, start=1):
        fold_macro_f1 = []
        fold_enrolled_f1 = []
        fold_enrolled_recall = []
        fit_times = []

        print(f"[SEARCH] {model_name} combo={combo_idx}/{total} params={params}", flush=True)
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train_text), start=1):
            X_tr = X_train.iloc[train_idx].copy()
            X_va = X_train.iloc[valid_idx].copy()
            y_tr_text = y_train_text.iloc[train_idx]
            y_va_text = y_train_text.iloc[valid_idx]

            model = make_model(model_name, params, n_jobs=n_jobs)
            start = time.perf_counter()
            if model_name == "lightgbm":
                model.fit(X_tr, y_tr_text, categorical_feature=cat_cols)
                pred_text = model.predict(X_va)
            else:
                y_tr_encoded = label_encoder.transform(y_tr_text)
                model.fit(X_tr, y_tr_encoded)
                pred_encoded = model.predict(X_va)
                pred_text = label_encoder.inverse_transform(pred_encoded.astype(int))
            fit_times.append(time.perf_counter() - start)

            macro_f1 = f1_score(y_va_text, pred_text, average="macro", zero_division=0)
            detail = class_metrics(y_va_text, pred_text)
            fold_macro_f1.append(macro_f1)
            fold_enrolled_f1.append(detail["enrolled_f1"])
            fold_enrolled_recall.append(detail["enrolled_recall"])
            print(
                f"[FOLD] {model_name} combo={combo_idx}/{total} fold={fold_idx}/{cv.get_n_splits()} "
                f"f1_macro={macro_f1:.4f} enrolled_f1={detail['enrolled_f1']:.4f}",
                flush=True,
            )

        record = {
            "model": model_name,
            "max_bin": int(params["max_bin"]),
            "column_fraction": float(params["column_fraction"]),
            "approx_effective_feature_count": int(math.ceil(n_features * float(params["column_fraction"]))),
            "n_features": n_features,
            "cv_mean_f1_macro": float(np.mean(fold_macro_f1)),
            "cv_std_f1_macro": float(np.std(fold_macro_f1)),
            "cv_mean_enrolled_f1": float(np.mean(fold_enrolled_f1)),
            "cv_std_enrolled_f1": float(np.std(fold_enrolled_f1)),
            "cv_mean_enrolled_recall": float(np.mean(fold_enrolled_recall)),
            "mean_fit_time_sec": float(np.mean(fit_times)),
        }
        records.append(record)
        save_metrics_table(records, METRICS_DIR / "hist_column_sampling_grid.csv")
    return records


def fit_and_evaluate_best(model_name, best_row, X_train, y_train_text, X_test, y_test_text, cat_cols, n_jobs):
    params = {"max_bin": int(best_row["max_bin"]), "column_fraction": float(best_row["column_fraction"])}
    model = make_model(model_name, params, n_jobs=n_jobs)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_text)

    if model_name == "lightgbm":
        model.fit(X_train, y_train_text, categorical_feature=cat_cols)
        pred_text = model.predict(X_test)
        prob = model.predict_proba(X_test)
    else:
        model.fit(X_train, y_train_encoded)
        pred_encoded = model.predict(X_test)
        pred_text = label_encoder.inverse_transform(pred_encoded.astype(int))
        prob = model.predict_proba(X_test)

    metrics = classification_metrics(y_test_text, pred_text, y_prob=prob, labels=LABELS)
    metrics.update(class_metrics(y_test_text, pred_text))
    metrics.update(
        {
            "model": model_name,
            "max_bin": params["max_bin"],
            "column_fraction": params["column_fraction"],
            "approx_effective_feature_count": int(best_row["approx_effective_feature_count"]),
            "cv_best_f1_macro": float(best_row["cv_mean_f1_macro"]),
            "cv_best_enrolled_f1": float(best_row["cv_mean_enrolled_f1"]),
        }
    )
    return metrics, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use a reduced smoke-test grid.")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("HIST_COLUMN_N_JOBS", "8")))
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, numeric_cols, cat_cols = load_data()
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    param_grid = make_param_grid(args.quick)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    all_grid_records = []
    for model_name in ["lightgbm", "xgboost_hist"]:
        all_grid_records.extend(
            evaluate_cv(model_name, X_train, y_train, cat_cols, cv, param_grid, n_jobs=args.n_jobs)
        )

    grid_df = pd.DataFrame(all_grid_records)
    grid_df = grid_df.sort_values(["model", "cv_mean_f1_macro", "cv_mean_enrolled_f1"], ascending=[True, False, False])
    grid_path = METRICS_DIR / "hist_column_sampling_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    metric_records = []
    best_params = {
        "metadata": {
            "cv": args.cv,
            "quick": args.quick,
            "n_jobs": args.n_jobs,
            "n_features": int(X_train.shape[1]),
            "categorical_columns": cat_cols,
            "numeric_or_ordinal_columns": numeric_cols,
            "selection_metric": "cv_mean_f1_macro, tie-break cv_mean_enrolled_f1",
            "max_bin_grid": [16, 32, 64, 128, 255],
            "column_fraction_grid": [0.60, 0.75, 0.90, 1.00],
        },
        "models": {},
    }
    for model_name in ["lightgbm", "xgboost_hist"]:
        model_grid = grid_df[grid_df["model"] == model_name].copy()
        best_row = model_grid.iloc[0].to_dict()
        metrics, params = fit_and_evaluate_best(
            model_name, best_row, X_train, y_train, X_test, y_test, cat_cols, n_jobs=args.n_jobs
        )
        metric_records.append(metrics)
        best_params["models"][model_name] = {
            "params": params,
            "approx_effective_feature_count": int(best_row["approx_effective_feature_count"]),
            "cv_mean_f1_macro": float(best_row["cv_mean_f1_macro"]),
            "cv_mean_enrolled_f1": float(best_row["cv_mean_enrolled_f1"]),
            "test_metrics": metrics,
        }

    metrics_path = METRICS_DIR / "hist_column_sampling_metrics.csv"
    params_path = METRICS_DIR / "hist_column_sampling_best_params.json"
    save_metrics_table(metric_records, metrics_path)
    params_path.write_text(json.dumps(json_safe(best_params), ensure_ascii=False, indent=2), encoding="utf-8")

    print(pd.DataFrame(metric_records).to_string(index=False))
    print(f"saved grid: {grid_path}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved params: {params_path}")


if __name__ == "__main__":
    main()
