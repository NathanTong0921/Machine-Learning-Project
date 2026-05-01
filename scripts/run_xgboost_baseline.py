"""Run XGBoost as a strong tabular boosting baseline.

Declaration:
- This script evaluates XGBoost on the student outcome dataset.
- It uses explicit categorical casting and XGBoost's categorical support.

Tips:
- Compare it against CatBoost, LightGBM, RF, soft voting, Multi-View Gated MLP, and TabPFN.
- On this dataset, stable multiclass probabilities are as important as raw accuracy.
"""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Compatibility shim for older xgboost releases under NumPy 2.x.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train_text = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL]).copy()
    y_test_text = test_df[TARGET_COL]

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_text)
    y_test = label_encoder.transform(y_test_text)

    cat_cols = [col for col in X_train.columns if col in CATEGORICAL_COLUMNS]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    y_pred_text = label_encoder.inverse_transform(y_pred)
    metrics = classification_metrics(y_test_text, y_pred_text, y_prob=y_prob, labels=LABELS)
    metrics["model"] = "xgboost"
    save_metrics_table([metrics], METRICS_DIR / "xgboost_metrics.csv")

    details = classification_details(y_test_text, y_pred_text, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / "cm_xgboost.png",
        "Confusion Matrix: XGBoost",
    )

    with open(METRICS_DIR / "xgboost_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "xgboost",
                "cat_features": cat_cols,
                "params": model.get_params(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"saved metrics: {METRICS_DIR / 'xgboost_metrics.csv'}")
    print(f"saved confusion matrix: {FIGURE_DIR / 'cm_xgboost.png'}")


if __name__ == "__main__":
    main()
