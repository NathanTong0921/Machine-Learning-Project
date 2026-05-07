#!/usr/bin/env python3
"""Build and save a fully one-hot encoded full dataset (no train/val/test split)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "data.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_fullyonehot"
OUTPUT_FILE = OUTPUT_DIR / "fullyonehot.csv"

TARGET_COL = "Target"

CATEGORICAL_COLUMNS = {
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
}


def build_fullyonehot() -> Path:
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()

    if TARGET_COL not in df.columns:
        raise KeyError(f"Missing target column: {TARGET_COL}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    target = df[TARGET_COL].copy().reset_index(drop=True)

    features = df[feature_cols].copy()
    categorical_cols = [c for c in feature_cols if c in CATEGORICAL_COLUMNS]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    if numeric_cols:
        numeric_fill = features[numeric_cols].median()
        features.loc[:, numeric_cols] = features[numeric_cols].fillna(numeric_fill)
    for col in categorical_cols:
        mode = features[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Missing"
        features.loc[:, col] = features[col].fillna(fill_value)

    encoded_features = pd.get_dummies(features, columns=categorical_cols, dtype=int)
    encoded = encoded_features.copy()
    encoded[TARGET_COL] = target

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    encoded.to_csv(OUTPUT_FILE, index=False)
    return OUTPUT_FILE


if __name__ == "__main__":
    output_path = build_fullyonehot()
    print(f"Saved: {output_path}")
    print(f"Rows: {pd.read_csv(output_path).shape[0]}")
