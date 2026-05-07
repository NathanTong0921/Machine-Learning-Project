"""Cross-split and cross-seed verification for classic boosting models.

Declaration:
- This script reuses the best parameters from
  `models/metrics/classic_boosting_best_params.json`.
- Each seed creates a fresh stratified train/test split and is also used as the
  model random state, so the check covers both cross-set and cross-seed drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, LABELS, METRICS_DIR, RANDOM_STATE, RAW_DATA_PATH, TARGET_COL
from src.data_utils import load_raw_data
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns
from src.visualization import save_metrics_table_figure


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


def make_model(name, seed):
    if name == "adaboost_stumps":
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(random_state=seed),
            random_state=seed,
        )
    if name == "gbdt_log_loss":
        return GradientBoostingClassifier(loss="log_loss", random_state=seed)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(loss="log_loss", early_stopping=False, random_state=seed)
    raise ValueError(f"Unknown classic boosting model: {name}")


def strip_model_prefix(params):
    return {key.removeprefix("model__"): value for key, value in params.items()}


def summarize(records):
    df = pd.DataFrame(records)
    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
        "dropout_f1",
        "enrolled_f1",
        "graduate_f1",
    ]
    summaries = []
    for model_name, group in df.groupby("model"):
        row = {"model": model_name, "n_splits": len(group)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
            row[f"{col}_min"] = float(group[col].min())
            row[f"{col}_max"] = float(group[col].max())
        summaries.append(row)
    return pd.DataFrame(summaries).sort_values("f1_macro_mean", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 62, 82, 102, 122])
    parser.add_argument("--test-size", type=float, default=0.30)
    args = parser.parse_args()

    best_path = METRICS_DIR / "classic_boosting_best_params.json"
    if not best_path.exists():
        raise FileNotFoundError("Missing classic_boosting_best_params.json. Run search_classic_boosting_models.py first.")

    with open(best_path, "r", encoding="utf-8") as f:
        best_payload = json.load(f)
    model_configs = best_payload["models"]

    df = load_raw_data(RAW_DATA_PATH)
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    numeric_features, categorical_features = split_feature_columns(X.columns)
    records = []

    for seed in args.seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            stratify=y,
            random_state=seed,
        )
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        for model_name, config in model_configs.items():
            print(f"[VERIFY] seed={seed} model={model_name}", flush=True)
            preprocessor = make_dense_preprocessor(numeric_features, categorical_features)
            model = make_model(model_name, seed)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )
            pipeline.set_params(**config["best_params"])
            fit_kwargs = {"model__sample_weight": sample_weight} if model_name in {"adaboost_stumps", "gbdt_log_loss"} else {}
            pipeline.fit(X_train, y_train, **fit_kwargs)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
            report = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)
            for label in LABELS:
                metrics[f"{label.lower()}_precision"] = report[label]["precision"]
                metrics[f"{label.lower()}_recall"] = report[label]["recall"]
                metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]
            metrics["model"] = model_name
            metrics["seed"] = seed
            metrics["test_size"] = args.test_size
            metrics["n_train"] = len(X_train)
            metrics["n_test"] = len(X_test)
            records.append(metrics)

    details_df = pd.DataFrame(records).sort_values(["model", "seed"])
    summary_df = summarize(records)

    details_path = METRICS_DIR / "classic_boosting_cross_seed_metrics.csv"
    summary_path = METRICS_DIR / "classic_boosting_cross_seed_summary.csv"
    save_metrics_table(details_df.to_dict(orient="records"), details_path)
    save_metrics_table(summary_df.to_dict(orient="records"), summary_path)

    summary_json = {
        "source_best_params": str(best_path),
        "raw_data": str(RAW_DATA_PATH),
        "seeds": args.seeds,
        "test_size": args.test_size,
        "selection_metric": "mean f1_macro across cross-set/cross-seed splits",
        "best_model_by_mean_f1_macro": summary_df.iloc[0].to_dict(),
    }
    with open(METRICS_DIR / "classic_boosting_cross_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    save_metrics_table_figure(
        summary_df[
            [
                "model",
                "f1_macro_mean",
                "f1_macro_std",
                "accuracy_mean",
                "roc_auc_ovr_macro_mean",
                "enrolled_f1_mean",
                "enrolled_f1_std",
            ]
        ],
        FIGURE_DIR / "classic_boosting_cross_seed_summary.png",
        "Classic Boosting Cross-Set / Cross-Seed Verification",
    )

    print("\n[SUMMARY]")
    print(summary_df.to_string(index=False))
    print(f"saved details: {details_path}")
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
