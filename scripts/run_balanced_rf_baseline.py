"""Run BalancedRandomForest as a class-imbalance-aware baseline.

Declaration:
- This script evaluates BalancedRandomForestClassifier on the student outcome dataset.
- It is intended as the first low-cost follow-up after the tuned CatBoost result.

Tips:
- Read macro-F1 and minority-class recall before accuracy.
- This model is especially relevant for the minority Enrolled class.
"""

from pathlib import Path
import json
import sys

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, FIGURE_DIR, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor
from src.visualization import save_confusion_matrix


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

    model = BalancedRandomForestClassifier(
        n_estimators=400,
        sampling_strategy="all",
        replacement=True,
        bootstrap=False,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = "balanced_random_forest"
    save_metrics_table([metrics], METRICS_DIR / "balanced_rf_metrics.csv")

    details = classification_details(y_test, y_pred, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / "cm_balanced_random_forest.png",
        "Confusion Matrix: Balanced Random Forest",
    )

    with open(METRICS_DIR / "balanced_rf_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": "balanced_random_forest",
                "params": model.get_params(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"saved metrics: {METRICS_DIR / 'balanced_rf_metrics.csv'}")
    print(f"saved confusion matrix: {FIGURE_DIR / 'cm_balanced_random_forest.png'}")


if __name__ == "__main__":
    main()
