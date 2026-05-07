"""Grid-search classic boosting models.

Declaration:
- This script covers classic boosting baselines not handled by the dedicated
  CatBoost, LightGBM, and XGBoost scripts.
- It evaluates AdaBoost, sklearn Gradient Boosting / GBM, and histogram-based
  gradient boosting on the three-class student outcome task.
- GBM and GBDT are treated as the same sklearn family here. LogitBoost-style
  multinomial log-loss boosting is represented by GradientBoostingClassifier
  with loss="log_loss"; BrownBoost and RankBoost are documented as not run
  because sklearn has no native implementation and RankBoost is a ranking
  method rather than a direct multiclass classifier.

Tips:
- Run `scripts/preprocessing/prepare_data.py` first.
- The search uses macro-F1 because `Enrolled` is the minority and hardest class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns
from src.visualization import save_confusion_matrix, save_confusion_matrix_grid, save_metrics_table_figure


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


def make_dense_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def make_model_specs():
    return {
        "adaboost_stumps": {
            "model": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__estimator__max_depth": [1, 2, 3],
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.1, 0.5, 1.0],
            },
            "fit_sample_weight": True,
            "description": "AdaBoost with shallow decision trees; misclassified samples receive larger weights.",
        },
        "gbdt_log_loss": {
            "model": GradientBoostingClassifier(
                loss="log_loss",
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
            "fit_sample_weight": True,
            "description": "Sklearn GBM/GBDT with multinomial log-loss; each stage fits the previous residual gradient.",
        },
        "hist_gradient_boosting": {
            "model": HistGradientBoostingClassifier(
                loss="log_loss",
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_iter": [200, 400],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [10, 20],
                "model__l2_regularization": [0.0, 0.1],
                "model__class_weight": ["balanced"],
            },
            "fit_sample_weight": False,
            "description": "Histogram-based gradient boosting; a fast sklearn GBDT variant.",
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
    preprocessor = make_dense_preprocessor(numeric_features, categorical_features)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    grid_records = []
    metric_records = []
    best_configs = {}
    cm_paths = []
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, spec in make_model_specs().items():
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
        fit_kwargs = {"model__sample_weight": sample_weight} if spec["fit_sample_weight"] else {}
        search.fit(X_train, y_train, **fit_kwargs)

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
        save_metrics_table(
            pd.concat(grid_records, ignore_index=True)
            .sort_values(["model", "rank_test_score"])
            .to_dict(orient="records"),
            METRICS_DIR / "classic_boosting_grid.csv",
        )
        save_metrics_table(
            pd.DataFrame(metric_records).sort_values("f1_macro", ascending=False).to_dict(orient="records"),
            METRICS_DIR / "classic_boosting_metrics.csv",
        )
        with open(METRICS_DIR / "classic_boosting_best_params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "selection_metric": "5-fold CV macro-F1",
                    "cv_splits": args.cv,
                    "random_state": RANDOM_STATE,
                    "preprocessing": "Appendix-aligned semantic split; dense one-hot for categorical features.",
                    "status": f"partial_through_{model_name}",
                    "models": best_configs,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(pd.DataFrame([metrics]).to_string(index=False), flush=True)

    grid_df = pd.concat(grid_records, ignore_index=True).sort_values(["model", "rank_test_score"])
    metrics_df = pd.DataFrame(metric_records).sort_values("f1_macro", ascending=False)

    save_metrics_table(grid_df.to_dict(orient="records"), METRICS_DIR / "classic_boosting_grid.csv")
    save_metrics_table(metrics_df.to_dict(orient="records"), METRICS_DIR / "classic_boosting_metrics.csv")

    with open(METRICS_DIR / "classic_boosting_best_params.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_metric": "5-fold CV macro-F1",
                "cv_splits": args.cv,
                "random_state": RANDOM_STATE,
                "preprocessing": "Appendix-aligned semantic split; dense one-hot for categorical features.",
                "not_run": {
                    "brownboost": "No native sklearn implementation; would require an external package or custom algorithm.",
                    "rankboost": "Ranking algorithm; not a direct fit for the three-class classification target.",
                },
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
        FIGURE_DIR / "classic_boosting_summary.png",
        "Classic Boosting Model Comparison",
    )
    save_confusion_matrix_grid(
        [path for path, _ in cm_paths],
        [title for _, title in cm_paths],
        FIGURE_DIR / "classic_boosting_confusion_grid.png",
        nrows=2,
        ncols=2,
    )

    print("\n[SUMMARY]")
    print(metrics_df.to_string(index=False))
    print(f"saved grid: {METRICS_DIR / 'classic_boosting_grid.csv'}")
    print(f"saved metrics: {METRICS_DIR / 'classic_boosting_metrics.csv'}")
    print(f"saved best params: {METRICS_DIR / 'classic_boosting_best_params.json'}")


if __name__ == "__main__":
    main()
