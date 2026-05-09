"""Reproduce the best single-split gbdt_log_loss row.

This script intentionally writes to a separate reproduction file and does not
overwrite `classic_boosting_cross_seed_metrics.csv`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, METRICS_DIR, RAW_DATA_PATH, TARGET_COL
from src.data_utils import load_raw_data
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns


OUTPUT_PATH = METRICS_DIR / "boost" / "gbdt_log_loss_reproduction.csv"
REFERENCE_PATH = METRICS_DIR / "boost" / "classic_boosting_cross_seed_metrics.csv"
BEST_PARAMS_PATH = METRICS_DIR / "boost" / "classic_boosting_best_params.json"


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


def load_best_params():
    payload = json.loads(BEST_PARAMS_PATH.read_text(encoding="utf-8"))
    return payload["models"]["gbdt_log_loss"]["best_params"]


def load_reference(seed: int, test_size: float):
    if not REFERENCE_PATH.exists():
        return {}
    reference_df = pd.read_csv(REFERENCE_PATH)
    rows = reference_df[
        (reference_df["model"] == "gbdt_log_loss")
        & (reference_df["seed"] == seed)
        & (reference_df["test_size"] == test_size)
    ]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--test-size", type=float, default=0.30)
    args = parser.parse_args()

    df = load_raw_data(RAW_DATA_PATH)
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )

    numeric_features, categorical_features = split_feature_columns(X.columns)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_dense_preprocessor(numeric_features, categorical_features)),
            ("model", GradientBoostingClassifier(loss="log_loss", random_state=args.seed)),
        ]
    )
    pipeline.set_params(**load_best_params())
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)
    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics.update(
        {
            "model": "gbdt_log_loss",
            "seed": args.seed,
            "test_size": args.test_size,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
    )

    reference = load_reference(args.seed, args.test_size)
    record = dict(metrics)
    for key in ["accuracy", "f1_macro", "f1_weighted", "roc_auc_ovr_macro"]:
        record[f"reference_{key}"] = reference.get(key)
        record[f"delta_{key}"] = metrics[key] - reference[key] if key in reference else None
    record["reference_source"] = str(REFERENCE_PATH.relative_to(PROJECT_ROOT)) if reference else ""

    save_metrics_table([record], OUTPUT_PATH)
    print(pd.DataFrame([record]).to_string(index=False))
    print(f"saved reproduction: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
