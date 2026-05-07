"""Run CatBoost as a strong categorical-aware baseline.

Declaration:
- This script adds CatBoost as a high-priority tabular baseline for the student outcome project.
- It uses the raw categorical columns directly instead of one-hot encoding.

Tips:
- CatBoost is especially relevant here because many integer-looking columns are actually categorical.
- Compare its macro-F1 directly with RF, soft voting, Multi-View Gated MLP, and TabPFN.
"""

from pathlib import Path
import json
import sys

import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL]).copy()
    y_test = test_df[TARGET_COL]

    cat_cols = [col for col in X_train.columns if col in CATEGORICAL_COLUMNS]
    cat_idx = [X_train.columns.get_loc(col) for col in cat_cols]

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        depth=6,
        learning_rate=0.03,
        iterations=500,
        verbose=0,
        random_seed=42,
    )

    model.fit(X_train, y_train, cat_features=cat_idx)
    y_pred = model.predict(X_test).ravel()
    y_prob = model.predict_proba(X_test)

    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = "catboost"
    save_metrics_table([metrics], METRICS_DIR / "catboost_metrics.csv")

    details = classification_details(y_test, y_pred, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / "cm_catboost.png",
        "Confusion Matrix: CatBoost",
    )

    with open(METRICS_DIR / "catboost_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "catboost",
                "cat_features": cat_cols,
                "params": model.get_params(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"saved metrics: {METRICS_DIR / 'catboost_metrics.csv'}")
    print(f"saved confusion matrix: {FIGURE_DIR / 'cm_catboost.png'}")


if __name__ == "__main__":
    main()
