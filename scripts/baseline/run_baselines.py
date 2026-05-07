"""Train baseline classifiers and save metrics.

Declaration:
- This script runs the required classical baselines for the course project.
- It is designed for reproducible py-based execution instead of notebooks.

Tips:
- Run `scripts/preprocessing/prepare_data.py` first.
- Start with these models before trying advanced tabular architectures.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.pipeline import Pipeline

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import get_baseline_models
from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor, split_feature_columns
from src.visualization import save_confusion_matrix


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    records = []
    for model_name, model in get_baseline_models(random_state=RANDOM_STATE).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

        metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
        metrics["model"] = model_name
        records.append(metrics)

        details = classification_details(y_test, y_pred, labels=LABELS)
        save_confusion_matrix(
            details["confusion_matrix"],
            LABELS,
            f"reports/figures/cm_{model_name}.png",
            f"Confusion Matrix: {model_name}",
        )
        print(f"\n[{model_name}]")
        print(details["report"])

    save_metrics_table(records, METRICS_DIR / "classification_metrics.csv")
    print(f"saved metrics: {METRICS_DIR / 'classification_metrics.csv'}")


if __name__ == "__main__":
    main()
