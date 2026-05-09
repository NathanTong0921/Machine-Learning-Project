"""Search RidgeClassifier hyperparameters as a regularized linear baseline.

RidgeClassifier is the classification analogue of ridge regression: it fits
one-vs-rest least-squares classifiers with L2 regularization controlled by
`alpha`. It is useful as a strong, low-variance linear baseline on the dense
one-hot tabular representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor, split_feature_columns


OUTPUT_DIR = METRICS_DIR / "baseline"
GRID_PATH = OUTPUT_DIR / "ridge_classifier_grid.csv"
METRICS_PATH = OUTPUT_DIR / "ridge_classifier_metrics.csv"
BEST_PATH = OUTPUT_DIR / "ridge_classifier_best.json"


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


def metric_row(model_name: str, set_name: str, y_true, y_pred, extra: dict[str, Any] | None = None):
    metrics = classification_metrics(y_true, y_pred, labels=LABELS)
    row = {
        "model": model_name,
        "Set": set_name,
        **metrics,
    }
    if extra:
        row.update(extra)
    return row


def evaluate_best(search, X_train, y_train, X_test, y_test, best_cv_row: pd.Series):
    pred_train = search.predict(X_train)
    pred_test = search.predict(X_test)
    y_entire = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)
    pred_entire = pd.Series(list(pred_train) + list(pred_test))
    extra = {
        "best_cv_f1_macro": float(best_cv_row["mean_test_score"]),
        "best_cv_std_f1_macro": float(best_cv_row["std_test_score"]),
        "alpha": search.best_params_.get("model__alpha"),
        "class_weight": search.best_params_.get("model__class_weight"),
        "solver": search.best_params_.get("model__solver"),
    }
    return [
        metric_row("ridge_classifier_tuned", "Training", y_train, pred_train, extra),
        metric_row("ridge_classifier_tuned", "Testing", y_test, pred_test, extra),
        metric_row("ridge_classifier_tuned", "Entire", y_entire, pred_entire, extra),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=64)
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", RidgeClassifier(random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["auto", "lsqr", "sparse_cg", "sag"],
    }
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=args.n_jobs,
        return_train_score=True,
        verbose=1,
    )
    print(f"[SEARCH] RidgeClassifier cv={args.cv} n_jobs={args.n_jobs}", flush=True)
    search.fit(X_train, y_train)

    grid_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    grid_df["model"] = "ridge_classifier"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_table(grid_df.to_dict("records"), GRID_PATH)

    metrics = evaluate_best(search, X_train, y_train, X_test, y_test, grid_df.iloc[0])
    save_metrics_table(metrics, METRICS_PATH)
    BEST_PATH.write_text(
        json.dumps(
            json_safe(
                {
                    "selection_metric": "cv_mean_f1_macro",
                    "cv_splits": args.cv,
                    "n_jobs": args.n_jobs,
                    "best_params": search.best_params_,
                    "best_cv_mean_f1_macro": grid_df.iloc[0]["mean_test_score"],
                    "best_cv_std_f1_macro": grid_df.iloc[0]["std_test_score"],
                    "metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
                    "grid_path": str(GRID_PATH.relative_to(PROJECT_ROOT)),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"saved grid: {GRID_PATH}")
    print(f"saved metrics: {METRICS_PATH}")
    print(f"saved best: {BEST_PATH}")


if __name__ == "__main__":
    main()
