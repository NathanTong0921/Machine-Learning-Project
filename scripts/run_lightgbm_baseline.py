"""Run LightGBM as a strong tabular boosting baseline.

Declaration:
- This script evaluates LightGBM on the student outcome dataset.
- It uses categorical-feature columns directly instead of one-hot encoding.

Tips:
- Compare it against CatBoost, RF, soft voting, Multi-View Gated MLP, and TabPFN.
- On this dataset, correct handling of categorical columns matters more than raw model complexity.
"""

from pathlib import Path
import json
import sys

import pandas as pd
from lightgbm import LGBMClassifier

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
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL]).copy()
    y_test = test_df[TARGET_COL]

    cat_cols = [col for col in X_train.columns if col in CATEGORICAL_COLUMNS]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=400,
        random_state=42,
        verbosity=-1,
    )

    model.fit(X_train, y_train, categorical_feature=cat_cols)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = "lightgbm"
    save_metrics_table([metrics], METRICS_DIR / "lightgbm_metrics.csv")

    details = classification_details(y_test, y_pred, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / "cm_lightgbm.png",
        "Confusion Matrix: LightGBM",
    )

    with open(METRICS_DIR / "lightgbm_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "lightgbm",
                "cat_features": cat_cols,
                "params": model.get_params(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"saved metrics: {METRICS_DIR / 'lightgbm_metrics.csv'}")
    print(f"saved confusion matrix: {FIGURE_DIR / 'cm_lightgbm.png'}")


if __name__ == "__main__":
    main()
