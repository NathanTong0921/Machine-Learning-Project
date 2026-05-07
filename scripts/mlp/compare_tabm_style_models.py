"""Compare the current best Multi-View Gated MLP with a TabM-style multi-expert MLP.

Declaration:
- This script evaluates whether a parameter-efficient multi-expert MLP can improve
  over the current best Multi-View Gated MLP on this dataset.
- It uses the same train/test split and the same evaluation metrics.

Tips:
- Read macro-F1 and AUC before accuracy.
- The TabM-style model here is a project-local implementation inspired by the TabM idea,
  not a literal reproduction of the official package.
"""

import json
from pathlib import Path
import sys

import pandas as pd
import torch

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import train_multiview_gated_mlp, train_tabm_style_multiexpert_mlp
from src.config import FIGURE_DIR, LABELS, METRICS_DIR, MODEL_DIR_MLP, PROCESSED_DIR
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

TABM_STYLE_CONFIG = {
    "seed": 62,
    "hidden_dim": 64,
    "dropout": 0.10,
    "num_experts": 8,
    "expert_hidden_dim": 96,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 80,
    "patience": 12,
}


def save_model_artifacts(result, checkpoint_name, metadata_name, gate_name=None):
    MODEL_DIR_MLP.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "metadata": {k: v for k, v in result["metadata"].items() if k != "history"},
        },
        MODEL_DIR_MLP / checkpoint_name,
    )
    with open(METRICS_DIR / metadata_name, "w", encoding="utf-8") as f:
        json.dump(result["metadata"], f, ensure_ascii=False, indent=2)
    if gate_name is not None and "gate_frame" in result:
        result["gate_frame"].groupby("true_label").mean(numeric_only=True).reindex(LABELS).reset_index().to_csv(
            METRICS_DIR / gate_name,
            index=False,
        )


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
    tabm_result = train_tabm_style_multiexpert_mlp(train_df, test_df, **TABM_STYLE_CONFIG)

    records = [
        evaluate_and_save("multiview_gated_mlp_best", mv_result, "cm_multiview_gated_mlp_best.png"),
        evaluate_and_save("tabm_style_multiexpert_mlp", tabm_result, "cm_tabm_style_multiexpert_mlp.png"),
    ]
    comparison_df = pd.DataFrame(records)
    output_csv = METRICS_DIR / "tabm_style_comparison.csv"
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
        FIGURE_DIR / "tabm_style_comparison.png",
        "TabM-Style Multi-Expert MLP Comparison",
    )
    save_confusion_matrix_grid(
        image_paths=[
            FIGURE_DIR / "cm_multiview_gated_mlp_best.png",
            FIGURE_DIR / "cm_tabm_style_multiexpert_mlp.png",
        ],
        titles=[
            "Best Multi-View Gated MLP",
            "TabM-Style Multi-Expert MLP",
        ],
        output_path=FIGURE_DIR / "tabm_style_confusion_grid.png",
        nrows=1,
        ncols=2,
    )

    save_model_artifacts(
        tabm_result,
        "tabm_style_multiexpert_mlp.pt",
        "tabm_style_multiexpert_mlp_metadata.json",
        gate_name="tabm_style_multiexpert_gate_summary.csv",
    )

    print(comparison_df.to_string(index=False))
    print(f"saved comparison csv: {output_csv}")
    print(f"saved comparison figure: {FIGURE_DIR / 'tabm_style_comparison.png'}")
    print(f"saved confusion grid: {FIGURE_DIR / 'tabm_style_confusion_grid.png'}")


if __name__ == "__main__":
    main()
