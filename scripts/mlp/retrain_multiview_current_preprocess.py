"""Retrain MultiView Gated MLP with the current Appendix-aligned preprocessing.

This script intentionally does not overwrite the old `multiview_gated_mlp_best`
checkpoint because that checkpoint was trained with a legacy categorical set
where `Application order` was one-hot encoded.
"""

import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import train_multiview_gated_mlp
from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, MODEL_DIR_MLP, PROCESSED_DIR
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix


CONFIG = {
    "seed": 62,
    "hidden_dim": 64,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
}

MODEL_NAME = "multiview_gated_mlp_current_preprocess"


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    result = train_multiview_gated_mlp(train_df, test_df, **CONFIG)
    metrics = classification_metrics(
        result["test_true_labels"],
        result["test_pred_labels"],
        y_prob=result["test_prob"],
        labels=LABELS,
    )
    metrics["model"] = MODEL_NAME

    details = classification_details(result["test_true_labels"], result["test_pred_labels"], labels=LABELS)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR_MLP.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    save_metrics_table([metrics], METRICS_DIR / f"{MODEL_NAME}_metrics.csv")
    pd.DataFrame(result["metadata"]["history"]).to_csv(METRICS_DIR / f"{MODEL_NAME}_training_history.csv", index=False)

    metadata = {
        **{k: v for k, v in result["metadata"].items() if k != "history"},
        "config": CONFIG,
        "model_name": MODEL_NAME,
        "categorical_columns": sorted(CATEGORICAL_COLUMNS),
        "preprocessing_note": "Current Appendix-aligned config keeps Application order numeric/ordinal.",
    }
    with open(METRICS_DIR / f"{MODEL_NAME}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "metadata": metadata,
        },
        MODEL_DIR_MLP / f"{MODEL_NAME}.pt",
    )

    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / f"cm_{MODEL_NAME}.png",
        f"Confusion Matrix: {MODEL_NAME}",
    )

    print(f"[{MODEL_NAME}]")
    print(details["report"])
    print(metrics)
    print(f"saved metrics: {METRICS_DIR / f'{MODEL_NAME}_metrics.csv'}")
    print(f"saved checkpoint: {MODEL_DIR_MLP / f'{MODEL_NAME}.pt'}")
    print(f"saved confusion matrix: {FIGURE_DIR / f'cm_{MODEL_NAME}.png'}")


if __name__ == "__main__":
    main()
