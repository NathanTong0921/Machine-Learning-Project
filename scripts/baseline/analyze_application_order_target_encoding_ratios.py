"""Analyze class ratios behind Application order OOF target encoding."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL


APP_ORDER_COL = "Application order"
OUTPUT_DIR = METRICS_DIR / "baseline"
OUT_ORDER_RATIOS = OUTPUT_DIR / "application_order_class_ratios_for_target_encoding.csv"
OUT_OOF_BY_TRUE_CLASS = OUTPUT_DIR / "application_order_oof_target_encoding_by_true_class.csv"
OUT_SUMMARY = OUTPUT_DIR / "application_order_target_encoding_balance_summary.csv"


def smoothed_class_rates(frame: pd.DataFrame, alpha: float):
    counts = frame[TARGET_COL].value_counts().reindex(LABELS, fill_value=0).astype(float)
    total = float(len(frame))
    return (counts + alpha) / (total + alpha * len(LABELS))


def add_oof_target_encoding(train_df: pd.DataFrame, n_splits=5, alpha=10.0):
    train_out = train_df.copy()
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

    train_out[columns] = train_out[columns].astype(float)
    return train_out, columns


def normalized_entropy(values):
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    entropy = -(values * np.log(values)).sum()
    return float(entropy / np.log(len(LABELS)))


def order_ratio_rows(train_df: pd.DataFrame, alpha=10.0):
    global_rates = smoothed_class_rates(train_df, alpha)
    rows = []
    for order_value, group in train_df.groupby(APP_ORDER_COL):
        counts = group[TARGET_COL].value_counts().reindex(LABELS, fill_value=0).astype(int)
        raw_rates = counts / len(group)
        smoothed = smoothed_class_rates(group, alpha)
        weight = len(group) / (len(group) + alpha)
        encoded = weight * smoothed + (1.0 - weight) * global_rates
        raw_values = raw_rates.reindex(LABELS).to_numpy()
        encoded_values = encoded.reindex(LABELS).to_numpy()
        rows.append(
            {
                "application_order": order_value,
                "support": len(group),
                "dropout_count": counts["Dropout"],
                "enrolled_count": counts["Enrolled"],
                "graduate_count": counts["Graduate"],
                "dropout_raw_ratio": raw_rates["Dropout"],
                "enrolled_raw_ratio": raw_rates["Enrolled"],
                "graduate_raw_ratio": raw_rates["Graduate"],
                "dropout_smoothed_te_ratio": encoded["Dropout"],
                "enrolled_smoothed_te_ratio": encoded["Enrolled"],
                "graduate_smoothed_te_ratio": encoded["Graduate"],
                "raw_max_minus_min": raw_values.max() - raw_values.min(),
                "smoothed_te_max_minus_min": encoded_values.max() - encoded_values.min(),
                "raw_entropy_norm": normalized_entropy(raw_values),
                "smoothed_te_entropy_norm": normalized_entropy(encoded_values),
                "majority_class_raw": LABELS[int(raw_values.argmax())],
                "majority_class_smoothed_te": LABELS[int(encoded_values.argmax())],
            }
        )
    return pd.DataFrame(rows).sort_values("application_order")


def oof_by_true_class_rows(oof_df: pd.DataFrame, te_columns):
    rows = []
    for label, group in oof_df.groupby(TARGET_COL):
        row = {"true_class": label, "support": len(group)}
        for col in te_columns:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_std"] = group[col].std(ddof=1)
        row["encoded_margin_to_second_mean"] = group[te_columns].apply(
            lambda values: values.nlargest(2).iloc[0] - values.nlargest(2).iloc[1],
            axis=1,
        ).mean()
        rows.append(row)
    label_order = {label: idx for idx, label in enumerate(LABELS)}
    return pd.DataFrame(rows).sort_values("true_class", key=lambda col: col.map(label_order))


def balance_summary(order_df: pd.DataFrame, oof_by_class: pd.DataFrame, train_df: pd.DataFrame):
    global_counts = train_df[TARGET_COL].value_counts().reindex(LABELS, fill_value=0)
    global_ratios = global_counts / len(train_df)
    return pd.DataFrame(
        [
            {
                "item": "global_train_class_ratio",
                "dropout": global_ratios["Dropout"],
                "enrolled": global_ratios["Enrolled"],
                "graduate": global_ratios["Graduate"],
                "note": "Baseline class prior in training split.",
            },
            {
                "item": "mean_raw_order_ratio",
                "dropout": order_df["dropout_raw_ratio"].mean(),
                "enrolled": order_df["enrolled_raw_ratio"].mean(),
                "graduate": order_df["graduate_raw_ratio"].mean(),
                "note": "Unweighted mean across Application order values.",
            },
            {
                "item": "mean_smoothed_te_ratio",
                "dropout": order_df["dropout_smoothed_te_ratio"].mean(),
                "enrolled": order_df["enrolled_smoothed_te_ratio"].mean(),
                "graduate": order_df["graduate_smoothed_te_ratio"].mean(),
                "note": "Unweighted mean after smoothing/shrinkage toward global prior.",
            },
            {
                "item": "mean_order_imbalance",
                "dropout": np.nan,
                "enrolled": np.nan,
                "graduate": np.nan,
                "note": (
                    f"Mean raw max-min={order_df['raw_max_minus_min'].mean():.4f}; "
                    f"mean smoothed max-min={order_df['smoothed_te_max_minus_min'].mean():.4f}."
                ),
            },
        ]
    )


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    order_df = order_ratio_rows(train_df)
    oof_df, te_columns = add_oof_target_encoding(train_df)
    by_class = oof_by_true_class_rows(oof_df, te_columns)
    summary = balance_summary(order_df, by_class, train_df)

    order_df.to_csv(OUT_ORDER_RATIOS, index=False)
    by_class.to_csv(OUT_OOF_BY_TRUE_CLASS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    print(order_df.to_string(index=False))
    print()
    print(by_class.to_string(index=False))
    print()
    print(summary.to_string(index=False))
    print(f"saved order ratios: {OUT_ORDER_RATIOS}")
    print(f"saved oof by true class: {OUT_OOF_BY_TRUE_CLASS}")
    print(f"saved summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
