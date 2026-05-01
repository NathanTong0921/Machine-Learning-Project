"""Train and compare the best Multi-View Gated MLP against FT-Transformer.

Declaration:
- This script provides a formal, same-split comparison between the current best
  Multi-View Gated MLP configuration and the new FT-Transformer implementation.
- It saves metrics, confusion matrices, model checkpoints, and a compact comparison table.

Tips:
- Run `scripts/prepare_data.py` first.
- Read `macro-F1` and `AUC` before reading raw accuracy.
"""

import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import train_ft_transformer, train_multiview_gated_mlp
from src.config import FIGURE_DIR, LABELS, METRICS_DIR, MODEL_DIR, PROCESSED_DIR
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_confusion_matrix_grid, save_metrics_table_figure


BEST_MV_CONFIG = {
    "seed": 62,
    "hidden_dim": 64,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
}

FT_CONFIG = {
    "seed": 62,
    "d_token": 48,
    "n_blocks": 2,
    "attention_n_heads": 4,
    "attention_dropout": 0.20,
    "ff_dropout": 0.20,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "batch_size": 128,
}


def save_model_artifacts(model_name, result, checkpoint_name, metadata_name):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "metadata": {k: v for k, v in result["metadata"].items() if k != "history"},
        },
        MODEL_DIR / checkpoint_name,
    )
    with open(METRICS_DIR / metadata_name, "w", encoding="utf-8") as f:
        json.dump(result["metadata"], f, ensure_ascii=False, indent=2)


def evaluate_and_record(model_name, result, cm_path):
    metrics = classification_metrics(
        result["test_true_labels"],
        result["test_pred_labels"],
        y_prob=result["test_prob"],
        labels=LABELS,
    )
    metrics["model"] = model_name

    details = classification_details(result["test_true_labels"], result["test_pred_labels"], labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        cm_path,
        f"Confusion Matrix: {model_name}",
    )
    return metrics


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    mv_result = train_multiview_gated_mlp(train_df, test_df, **BEST_MV_CONFIG)
    ft_result = train_ft_transformer(train_df, test_df, **FT_CONFIG)

    mv_metrics = evaluate_and_record(
        "multiview_gated_mlp_best",
        mv_result,
        FIGURE_DIR / "cm_multiview_gated_mlp_best.png",
    )
    ft_metrics = evaluate_and_record(
        "ft_transformer",
        ft_result,
        FIGURE_DIR / "cm_ft_transformer.png",
    )

    comparison_df = pd.DataFrame([mv_metrics, ft_metrics])
    output_csv = METRICS_DIR / "advanced_model_comparison.csv"
    save_metrics_table(comparison_df.to_dict(orient="records"), output_csv)
    save_metrics_table_figure(
        comparison_df[
            [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
            ]
        ],
        FIGURE_DIR / "advanced_model_comparison.png",
        "Advanced Model Comparison",
    )

    save_confusion_matrix_grid(
        image_paths=[
            FIGURE_DIR / "cm_multiview_gated_mlp_best.png",
            FIGURE_DIR / "cm_ft_transformer.png",
        ],
        titles=[
            "Best Multi-View Gated MLP",
            "FT-Transformer",
        ],
        output_path=FIGURE_DIR / "advanced_model_confusion_grid.png",
        nrows=1,
        ncols=2,
    )

    save_model_artifacts(
        "multiview_gated_mlp_best",
        mv_result,
        "multiview_gated_mlp_best.pt",
        "multiview_gated_mlp_best_metadata.json",
    )
    save_model_artifacts(
        "ft_transformer",
        ft_result,
        "ft_transformer.pt",
        "ft_transformer_metadata.json",
    )

    print(comparison_df.to_string(index=False))
    print(f"saved comparison csv: {output_csv}")
    print(f"saved comparison figure: {FIGURE_DIR / 'advanced_model_comparison.png'}")
    print(f"saved confusion grid: {FIGURE_DIR / 'advanced_model_confusion_grid.png'}")


if __name__ == "__main__":
    main()
