"""Grid-search bagging-style decision tree models.

Declaration:
- This script focuses on tree bagging methods for the open-ended exploration.
- It compares native decision-tree bagging, pasting-style decision-tree ensembles
  (sampling without replacement), and Extra Trees.
- The target is three-class classification, so "pasting regression" is adapted as
  pasting-style classification with `bootstrap=False`.

Tips:
- Run `scripts/preprocessing/prepare_data.py` first.
- The grid uses macro-F1 as the selection metric because `Enrolled` is the
  minority and hardest class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor, split_feature_columns
from src.visualization import save_confusion_matrix, save_confusion_matrix_grid, save_metrics_table_figure


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def make_model_specs(n_jobs: int):
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")

    bagging = BaggingClassifier(
        estimator=tree,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    pasting = BaggingClassifier(
        estimator=tree,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    extra_trees = ExtraTreesClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    bagging_grid = {
        "model__n_estimators": [200, 500],
        "model__max_samples": [0.65, 0.8, 1.0],
        "model__max_features": [0.7, 1.0],
        "model__estimator__max_depth": [6, 10, None],
        "model__estimator__min_samples_leaf": [1, 3, 5],
    }
    extra_trees_grid = {
        "model__n_estimators": [400, 800],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", 0.5, 1.0],
        "model__class_weight": ["balanced", "balanced_subsample"],
    }

    return {
        "bagging_decision_tree": {
            "model": bagging,
            "param_grid": bagging_grid,
            "description": "Bootstrap-sampled decision-tree bagging.",
        },
        "pasting_decision_tree": {
            "model": pasting,
            "param_grid": bagging_grid,
            "description": "Decision-tree bagging without replacement (`bootstrap=False`).",
        },
        "extra_trees": {
            "model": extra_trees,
            "param_grid": extra_trees_grid,
            "description": "Extremely randomized trees with random split thresholds.",
        },
    }


def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = name

    details = classification_details(y_test, y_pred, labels=LABELS)
    report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    for label in LABELS:
        metrics[f"{label.lower()}_precision"] = report[label]["precision"]
        metrics[f"{label.lower()}_recall"] = report[label]["recall"]
        metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]

    cm_path = FIGURE_DIR / f"cm_{name}.png"
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        cm_path,
        f"Confusion Matrix: {name}",
    )
    return metrics, cm_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true", help="Use a smaller smoke-test grid.")
    args = parser.parse_args()

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    grid_records = []
    metric_records = []
    best_configs = {}
    cm_paths = []

    for model_name, spec in make_model_specs(args.n_jobs).items():
        print(f"[SEARCH] {model_name}: {spec['description']}", flush=True)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", spec["model"]),
            ]
        )
        param_grid = spec["param_grid"]
        if args.quick:
            param_grid = {key: values[:1] for key, values in param_grid.items()}

        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=args.n_jobs,
            verbose=2,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train)

        cv_results = pd.DataFrame(search.cv_results_)
        keep_cols = [
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score",
            "rank_test_score",
            "mean_fit_time",
            "params",
        ]
        model_grid = cv_results[keep_cols].copy()
        model_grid.insert(0, "model", model_name)
        grid_records.append(model_grid)

        metrics, cm_path = evaluate_model(model_name, search.best_estimator_, X_test, y_test)
        metrics["cv_best_score_f1_macro"] = float(search.best_score_)
        metric_records.append(metrics)
        cm_paths.append((cm_path, model_name))

        best_configs[model_name] = {
            "description": spec["description"],
            "best_cv_f1_macro": float(search.best_score_),
            "best_params": json_safe(search.best_params_),
            "test_metrics": json_safe(metrics),
        }
        print(pd.DataFrame([metrics]).to_string(index=False), flush=True)

    grid_df = pd.concat(grid_records, ignore_index=True).sort_values(["model", "rank_test_score"])
    metrics_df = pd.DataFrame(metric_records).sort_values("f1_macro", ascending=False)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_table(grid_df.to_dict(orient="records"), METRICS_DIR / "bagging_tree_grid.csv")
    save_metrics_table(metrics_df.to_dict(orient="records"), METRICS_DIR / "bagging_tree_metrics.csv")

    with open(METRICS_DIR / "bagging_tree_best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_metric": "5-fold CV macro-F1",
                "cv_splits": args.cv,
                "random_state": RANDOM_STATE,
                "models": best_configs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_metrics_table_figure(
        metrics_df[
            [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
                "cv_best_score_f1_macro",
            ]
        ],
        FIGURE_DIR / "bagging_tree_summary.png",
        "Bagging Tree Model Comparison",
    )
    save_confusion_matrix_grid(
        [path for path, _ in cm_paths],
        [title for _, title in cm_paths],
        FIGURE_DIR / "bagging_tree_confusion_grid.png",
        nrows=2,
        ncols=2,
    )

    print("\n[SUMMARY]")
    print(metrics_df.to_string(index=False))
    print(f"saved grid: {METRICS_DIR / 'bagging_tree_grid.csv'}")
    print(f"saved metrics: {METRICS_DIR / 'bagging_tree_metrics.csv'}")
    print(f"saved best params: {METRICS_DIR / 'bagging_tree_best_params.json'}")


if __name__ == "__main__":
    main()
