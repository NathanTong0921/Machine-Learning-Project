"""Entry point for advanced tabular models.

Declaration:
- This script is reserved for non-standard models such as Multi-View Gated MLP,
  FT-Transformer, or TabNet.
- The current implementation focuses on a Multi-View Gated MLP because the dataset
  naturally splits into enrollment, family, first-semester, and second-semester views.

Tips:
- Only implement one advanced model first.
- Keep the same split and main metrics as the baselines for fair comparison.
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
from src.config import FIGURE_DIR, LABELS, METRICS_DIR, MODEL_DIR, PROCESSED_DIR
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    result = train_multiview_gated_mlp(train_df, test_df)

    metrics = classification_metrics(
        result["test_true_labels"],
        result["test_pred_labels"],
        y_prob=result["test_prob"],
        labels=LABELS,
    )
    metrics["model"] = "multiview_gated_mlp"
    save_metrics_table([metrics], METRICS_DIR / "advanced_metrics.csv")

    details = classification_details(result["test_true_labels"], result["test_pred_labels"], labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / "cm_multiview_gated_mlp.png",
        "Confusion Matrix: Multi-View Gated MLP",
    )

    gate_summary = (
        result["gate_frame"]
        .groupby("true_label")
        .mean(numeric_only=True)
        .reindex(LABELS)
        .reset_index()
    )
    gate_summary.to_csv(METRICS_DIR / "multiview_gate_summary.csv", index=False)
    pd.DataFrame(result["metadata"]["history"]).to_csv(METRICS_DIR / "advanced_training_history.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "metadata": {k: v for k, v in result["metadata"].items() if k != "history"},
        },
        MODEL_DIR / "multiview_gated_mlp.pt",
    )
    with open(METRICS_DIR / "advanced_metadata.json", "w", encoding="utf-8") as f:
        json.dump(result["metadata"], f, ensure_ascii=False, indent=2)

    print("[multiview_gated_mlp]")
    print(details["report"])
    print(f"saved metrics: {METRICS_DIR / 'advanced_metrics.csv'}")
    print(f"saved gate summary: {METRICS_DIR / 'multiview_gate_summary.csv'}")
    print(f"saved history: {METRICS_DIR / 'advanced_training_history.csv'}")
    print(f"saved model: {MODEL_DIR / 'multiview_gated_mlp.pt'}")


if __name__ == "__main__":
    main()
