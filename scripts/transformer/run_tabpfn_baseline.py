"""Run TabPFN as an external fancy baseline on the student outcome dataset.

Declaration:
- This script uses the official TabPFN package as an external tabular foundation-model baseline.
- It is intentionally minimal: fit, predict, evaluate, and save artifacts.

Tips:
- TabPFN may require first-run authentication and model-weight download.
- GPU is recommended and available in the current environment.
"""

from pathlib import Path
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix


def main():
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    from tabpfn import TabPFNClassifier

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    clf = TabPFNClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = "tabpfn"
    save_metrics_table([metrics], METRICS_DIR / "tabpfn_metrics.csv")

    details = classification_details(y_test, y_pred, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / "cm_tabpfn.png",
        "Confusion Matrix: TabPFN",
    )

    metadata = {
        "model": "tabpfn",
        "n_train": len(train_df),
        "n_test": len(test_df),
        "features": list(X_train.columns),
    }
    with open(METRICS_DIR / "tabpfn_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"saved metrics: {METRICS_DIR / 'tabpfn_metrics.csv'}")
    print(f"saved confusion matrix: {FIGURE_DIR / 'cm_tabpfn.png'}")


if __name__ == "__main__":
    main()
