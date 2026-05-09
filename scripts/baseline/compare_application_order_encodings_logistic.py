"""Compare Application order encodings for L2 Logistic Regression.

This mirrors the MLP Application-order encoding check with a linear model.
The goal is to test whether the ordinal encoding assumption also affects
Logistic Regression, whose decision boundary is directly tied to the input
feature geometry.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor, split_feature_columns


APP_ORDER_COL = "Application order"
OUTPUT_DIR = METRICS_DIR / "baseline"
OUT_METRICS = OUTPUT_DIR / "logistic_application_order_encoding_metrics.csv"
OUT_GRID = OUTPUT_DIR / "logistic_application_order_encoding_grid.csv"
OUT_BEST = OUTPUT_DIR / "logistic_application_order_encoding_best.json"

C_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def add_thermometer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_out = train_df.copy()
    test_out = test_df.copy()
    columns = []
    for threshold in range(1, 10):
        col = f"Application order >= {threshold}"
        train_out[col] = (train_out[APP_ORDER_COL] >= threshold).astype(float)
        test_out[col] = (test_out[APP_ORDER_COL] >= threshold).astype(float)
        columns.append(col)
    train_out = train_out.drop(columns=[APP_ORDER_COL])
    test_out = test_out.drop(columns=[APP_ORDER_COL])
    return train_out, test_out, columns


def smoothed_class_rates(frame: pd.DataFrame, alpha: float):
    counts = frame[TARGET_COL].value_counts().reindex(LABELS, fill_value=0).astype(float)
    total = float(len(frame))
    return (counts + alpha) / (total + alpha * len(LABELS))


def add_oof_target_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits=5, alpha=10.0):
    train_out = train_df.copy()
    test_out = test_df.copy()
    columns = [f"Application order te {label.lower()}" for label in LABELS]
    for col in columns:
        train_out[col] = np.nan

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fit_idx, valid_idx in cv.split(train_df, train_df[TARGET_COL]):
        fold_train = train_df.iloc[fit_idx]
        fold_global = smoothed_class_rates(fold_train, alpha)
        maps = {}
        for order_value, group in fold_train.groupby(APP_ORDER_COL):
            rates = smoothed_class_rates(group, alpha)
            weight = len(group) / (len(group) + alpha)
            maps[order_value] = weight * rates + (1.0 - weight) * fold_global

        for row_idx in valid_idx:
            order_value = train_df.iloc[row_idx][APP_ORDER_COL]
            rates = maps.get(order_value, fold_global)
            for label, col in zip(LABELS, columns):
                train_out.iat[row_idx, train_out.columns.get_loc(col)] = rates[label]

    global_rates = smoothed_class_rates(train_df, alpha)
    full_maps = {}
    for order_value, group in train_df.groupby(APP_ORDER_COL):
        rates = smoothed_class_rates(group, alpha)
        weight = len(group) / (len(group) + alpha)
        full_maps[order_value] = weight * rates + (1.0 - weight) * global_rates

    for col in columns:
        test_out[col] = np.nan
    for row_idx, order_value in enumerate(test_df[APP_ORDER_COL]):
        rates = full_maps.get(order_value, global_rates)
        for label, col in zip(LABELS, columns):
            test_out.iat[row_idx, test_out.columns.get_loc(col)] = rates[label]

    train_out[columns] = train_out[columns].astype(float)
    test_out[columns] = test_out[columns].astype(float)
    train_out = train_out.drop(columns=[APP_ORDER_COL])
    test_out = test_out.drop(columns=[APP_ORDER_COL])
    return train_out, test_out, columns


def make_pipeline(numeric_features, categorical_features):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    max_iter=6000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_set(model, X, y, set_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    metrics = classification_metrics(y, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["set"] = set_name
    return metrics


def run_variant(name, train_df, test_df, numeric_features, categorical_features, note):
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    X_entire = pd.concat([X_train, X_test], axis=0)
    y_entire = pd.concat([y_train, y_test], axis=0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        make_pipeline(numeric_features, categorical_features),
        param_grid={"model__C": C_GRID},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    grid.fit(X_train, y_train)

    records = []
    for set_name, X, y in [("Training", X_train, y_train), ("Testing", X_test, y_test), ("Entire", X_entire, y_entire)]:
        row = evaluate_set(grid.best_estimator_, X, y, set_name)
        row["model"] = name
        row["type"] = "L2 LogisticRegression"
        row["encoding_note"] = note
        row["best_C"] = grid.best_params_["model__C"]
        row["cv_f1_macro_mean"] = grid.best_score_
        row["n_numeric_features"] = len(numeric_features)
        row["n_categorical_features"] = len(categorical_features)
        records.append(row)

    grid_rows = []
    for idx, params in enumerate(grid.cv_results_["params"]):
        grid_rows.append(
            {
                "model": name,
                "C": params["model__C"],
                "mean_train_f1_macro": grid.cv_results_["mean_train_score"][idx],
                "std_train_f1_macro": grid.cv_results_["std_train_score"][idx],
                "mean_cv_f1_macro": grid.cv_results_["mean_test_score"][idx],
                "std_cv_f1_macro": grid.cv_results_["std_test_score"][idx],
                "rank": grid.cv_results_["rank_test_score"][idx],
                "encoding_note": note,
            }
        )
    return records, grid_rows


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = []

    X_train = train_df.drop(columns=[TARGET_COL])
    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    variants.append(
        (
            "logistic_app_order_scalar",
            train_df,
            test_df,
            numeric_features,
            categorical_features,
            "Current preprocessing: Application order is a scaled numeric/ordinal scalar.",
        )
    )

    onehot_numeric = [col for col in numeric_features if col != APP_ORDER_COL]
    onehot_categorical = list(categorical_features) + [APP_ORDER_COL]
    variants.append(
        (
            "logistic_app_order_onehot",
            train_df,
            test_df,
            onehot_numeric,
            onehot_categorical,
            "Treat Application order as a nominal category and one-hot encode it.",
        )
    )

    thermo_train, thermo_test, thermo_cols = add_thermometer_features(train_df, test_df)
    thermo_numeric, thermo_categorical = split_feature_columns(thermo_train.drop(columns=[TARGET_COL]).columns)
    variants.append(
        (
            "logistic_app_order_thermometer",
            thermo_train,
            thermo_test,
            thermo_numeric,
            thermo_categorical,
            "Replace Application order with nine cumulative threshold indicators.",
        )
    )

    te_train, te_test, te_cols = add_oof_target_encoding(train_df, test_df)
    te_numeric, te_categorical = split_feature_columns(te_train.drop(columns=[TARGET_COL]).columns)
    variants.append(
        (
            "logistic_app_order_oof_target_encoding",
            te_train,
            te_test,
            te_numeric,
            te_categorical,
            "Replace Application order with 5-fold OOF smoothed class-rate target encoding.",
        )
    )

    metric_records = []
    grid_records = []
    for args in variants:
        records, grid_rows = run_variant(*args)
        metric_records.extend(records)
        grid_records.extend(grid_rows)
        test_row = next(row for row in records if row["set"] == "Testing")
        print(
            f"{test_row['model']}: test macro-F1={test_row['f1_macro']:.4f}, "
            f"acc={test_row['accuracy']:.4f}, AUC={test_row['roc_auc_ovr_macro']:.4f}, "
            f"C={test_row['best_C']}"
        )

    save_metrics_table(metric_records, OUT_METRICS)
    pd.DataFrame(grid_records).to_csv(OUT_GRID, index=False)

    test_records = [row for row in metric_records if row["set"] == "Testing"]
    best = max(test_records, key=lambda row: row["f1_macro"])
    with open(OUT_BEST, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f"saved metrics: {OUT_METRICS}")
    print(f"saved grid: {OUT_GRID}")
    print(f"saved best: {OUT_BEST}")


if __name__ == "__main__":
    main()
