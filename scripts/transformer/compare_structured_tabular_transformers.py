"""Compare tuned FT-Transformer, view-aware FT, and best Multi-View Gated MLP.

Declaration:
- This script evaluates whether changing FT tokenization and making the model
  more structure-aware is better than simply scaling a standard FT-Transformer.
- It uses the same train/test split for all compared models.

Tips:
- Read macro-F1 first.
- If the view-aware variant closes the gap to the Multi-View Gated MLP, that is
  stronger evidence than further enlarging a standard FT.
"""

import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import (
    train_ft_transformer,
    train_multiview_gated_mlp,
    train_view_aware_ft_transformer,
)
from src.config import FIGURE_DIR, LABELS, METRICS_DIR, MODEL_DIR_TRANSFORMER, PROCESSED_DIR
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

TUNED_FT_CONFIG = {
    "seed": 62,
    "d_token": 48,
    "n_blocks": 2,
    "attention_n_heads": 4,
    "attention_dropout": 0.20,
    "ff_dropout": 0.20,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 60,
    "patience": 10,
}

VIEW_AWARE_FT_CONFIG = {
    "seed": 62,
    "d_token": 48,
    "n_blocks": 2,
    "attention_n_heads": 4,
    "attention_dropout": 0.20,
    "ff_dropout": 0.20,
    "n_num_frequencies": 8,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 60,
    "patience": 10,
}


def save_model_artifacts(result, checkpoint_name, metadata_name):
    MODEL_DIR_TRANSFORMER.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "metadata": {k: v for k, v in result["metadata"].items() if k != "history"},
        },
        MODEL_DIR_TRANSFORMER / checkpoint_name,
    )
    with open(METRICS_DIR / metadata_name, "w", encoding="utf-8") as f:
        json.dump(result["metadata"], f, ensure_ascii=False, indent=2)


def evaluate_and_save(model_name, result, cm_name):
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
        FIGURE_DIR / cm_name,
        f"Confusion Matrix: {model_name}",
    )
    return metrics


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    mv_result = train_multiview_gated_mlp(train_df, test_df, **BEST_MV_CONFIG)
    ft_result = train_ft_transformer(train_df, test_df, **TUNED_FT_CONFIG)
    structured_ft_result = train_view_aware_ft_transformer(train_df, test_df, **VIEW_AWARE_FT_CONFIG)

    records = [
        evaluate_and_save("multiview_gated_mlp_best", mv_result, "cm_multiview_gated_mlp_best.png"),
        evaluate_and_save("ft_transformer_tuned", ft_result, "cm_ft_transformer_tuned.png"),
        evaluate_and_save("view_aware_ft_transformer", structured_ft_result, "cm_view_aware_ft_transformer.png"),
    ]
    comparison_df = pd.DataFrame(records)
    output_csv = METRICS_DIR / "structured_transformer_comparison.csv"
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
        FIGURE_DIR / "structured_transformer_comparison.png",
        "Structured Transformer Comparison",
    )

    save_confusion_matrix_grid(
        image_paths=[
            FIGURE_DIR / "cm_multiview_gated_mlp_best.png",
            FIGURE_DIR / "cm_ft_transformer_tuned.png",
            FIGURE_DIR / "cm_view_aware_ft_transformer.png",
        ],
        titles=[
            "Best Multi-View Gated MLP",
            "FT-Transformer (Tuned)",
            "View-Aware FT-Transformer",
        ],
        output_path=FIGURE_DIR / "structured_transformer_confusion_grid.png",
        nrows=1,
        ncols=3,
    )

    save_model_artifacts(
        structured_ft_result,
        "view_aware_ft_transformer.pt",
        "view_aware_ft_transformer_metadata.json",
    )

    print(comparison_df.to_string(index=False))
    print(f"saved comparison csv: {output_csv}")
    print(f"saved comparison figure: {FIGURE_DIR / 'structured_transformer_comparison.png'}")
    print(f"saved confusion grid: {FIGURE_DIR / 'structured_transformer_confusion_grid.png'}")


if __name__ == "__main__":
    main()
