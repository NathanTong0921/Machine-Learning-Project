"""Check whether SVM-style noise-feature removal helps Logistic Regression.

This mirrors the final-report boost denoise check:
- merge processed train/test,
- run stratified splits over multiple seeds,
- compare original dense preprocessing with the same six dropped columns.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor, split_feature_columns


OUTPUT_DIR = METRICS_DIR / "baseline"
SEED_LEVEL_PATH = OUTPUT_DIR / "logistic_denoise_cross_seed_metrics.csv"
SUMMARY_PATH = OUTPUT_DIR / "logistic_denoise_cross_seed_summary.csv"

SEEDS = [42, 62, 82]
TEST_SIZE = 0.3
C_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

NOISE_DROP_COLUMNS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (without evaluations)",
    "International",
    "Educational special needs",
]

SUMMARY_METRICS = [
    "accuracy",
    "f1_macro",
    "roc_auc_ovr_macro",
    "enrolled_recall",
    "enrolled_f1",
    "dropout_f1",
    "graduate_f1",
]


def make_pipeline(numeric_features: list[str], categorical_features: list[str], seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    max_iter=6000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_set(model: Pipeline, X: pd.DataFrame, y: pd.Series, set_name: str) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    metrics = classification_metrics(y, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["set"] = set_name
    return metrics


def run_variant(
    variant: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    seed: int,
) -> list[dict]:
    dropped_count = 0
    if variant == "drop_noise":
        drop_cols = [col for col in NOISE_DROP_COLUMNS if col in X_train.columns]
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=drop_cols)
        dropped_count = len(drop_cols)
    elif variant != "original":
        raise ValueError(f"unknown variant: {variant}")

    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        estimator=make_pipeline(numeric_features, categorical_features, seed),
        param_grid={"model__C": C_GRID},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    X_entire = pd.concat([X_train, X_test], axis=0)
    y_entire = pd.concat([y_train, y_test], axis=0)

    rows = []
    for set_name, X, y in [("Training", X_train, y_train), ("Testing", X_test, y_test), ("Entire", X_entire, y_entire)]:
        row = evaluate_set(grid.best_estimator_, X, y, set_name)
        row.update(
            {
                "model": f"logistic_regression_{variant}",
                "variant": variant,
                "seed": seed,
                "best_C": grid.best_params_["model__C"],
                "cv_f1_macro_mean": grid.best_score_,
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "dropped_columns": dropped_count,
            }
        )
        rows.append(row)
    return rows


def build_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    test_df = seed_df[seed_df["set"].eq("Testing")].copy()
    rows = []
    for variant, group in test_df.groupby("variant", sort=False):
        row = {"model": f"logistic_regression_{variant}", "variant": variant, "n_seeds": len(group)}
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        row["best_C_values"] = ",".join(str(value) for value in group["best_C"].tolist())
        rows.append(row)

    summary = pd.DataFrame(rows)
    if {"original", "drop_noise"}.issubset(set(summary["variant"])):
        original = summary[summary["variant"].eq("original")].iloc[0]
        drop_noise = summary[summary["variant"].eq("drop_noise")].iloc[0]
        delta = {"model": "logistic_regression_delta", "variant": "drop_noise - original", "n_seeds": len(SEEDS)}
        for metric in SUMMARY_METRICS:
            delta[f"{metric}_mean"] = drop_noise[f"{metric}_mean"] - original[f"{metric}_mean"]
            delta[f"{metric}_std"] = 0.0
        delta["best_C_values"] = ""
        summary = pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_df = pd.concat(
        [pd.read_csv(PROCESSED_DIR / "train.csv"), pd.read_csv(PROCESSED_DIR / "test.csv")],
        axis=0,
        ignore_index=True,
    )

    X = full_df.drop(columns=[TARGET_COL])
    y = full_df[TARGET_COL]

    rows = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=seed,
        )
        for variant in ["original", "drop_noise"]:
            rows.extend(run_variant(variant, X_train, X_test, y_train, y_test, seed))

    seed_df = pd.DataFrame(rows)
    save_metrics_table(rows, SEED_LEVEL_PATH)
    summary = build_summary(seed_df)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"saved seed-level metrics: {SEED_LEVEL_PATH}")
    print(f"saved summary: {SUMMARY_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
