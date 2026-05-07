"""Compare cost-sensitive Multi-View Gated MLP variants.

Declaration:
- This script keeps the current best Multi-View Gated MLP configuration fixed.
- It only changes class-weight multipliers, mainly for the Enrolled class.
- The goal is to test whether a cost-sensitive loss improves macro-F1 or Enrolled recall/F1.
"""

import json
from pathlib import Path
import sys

import pandas as pd
import torch
from sklearn.metrics import classification_report

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import train_multiview_gated_mlp
from src.config import FIGURE_DIR, LABELS, METRICS_DIR, MODEL_DIR, PROCESSED_DIR
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_confusion_matrix_grid, save_metrics_table_figure


BASE_CONFIG = {
    "seed": 62,
    "hidden_dim": 64,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
}

VARIANTS = [
    ("baseline_recheck", {}),
    ("enrolled_x1_15", {"Enrolled": 1.15}),
    ("enrolled_x1_30", {"Enrolled": 1.30}),
    ("enrolled_x1_50", {"Enrolled": 1.50}),
    ("enrolled_x1_30_dropout_015", {"Enrolled": 1.30}, {"dropout": 0.15}),
]


def save_model_artifacts(result, checkpoint_name, metadata_name):
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


def evaluate_variant(model_name, result, cm_path):
    metrics = classification_metrics(
        result["test_true_labels"],
        result["test_pred_labels"],
        y_prob=result["test_prob"],
        labels=LABELS,
    )
    details = classification_details(result["test_true_labels"], result["test_pred_labels"], labels=LABELS)
    report = classification_report(
        result["test_true_labels"],
        result["test_pred_labels"],
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics["model"] = model_name
    for label in LABELS:
        metrics[f"{label.lower()}_precision"] = report[label]["precision"]
        metrics[f"{label.lower()}_recall"] = report[label]["recall"]
        metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]

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
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    records = []
    cm_paths = []
    cm_titles = []

    for variant in VARIANTS:
        name = variant[0]
        class_weight_multipliers = variant[1]
        config_overrides = variant[2] if len(variant) > 2 else {}
        config = {**BASE_CONFIG, **config_overrides}

        result = train_multiview_gated_mlp(
            train_df,
            test_df,
            class_weight_multipliers=class_weight_multipliers,
            **config,
        )
        result["metadata"]["variant"] = name
        result["metadata"]["config"] = config

        cm_path = FIGURE_DIR / f"cm_multiview_cost_sensitive_{name}.png"
        metrics = evaluate_variant(f"multiview_cost_sensitive_{name}", result, cm_path)
        metrics["class_weight_multipliers"] = json.dumps(class_weight_multipliers, sort_keys=True)
        records.append(metrics)
        cm_paths.append(cm_path)
        cm_titles.append(name)

        save_model_artifacts(
            result,
            f"multiview_cost_sensitive_{name}.pt",
            f"multiview_cost_sensitive_{name}_metadata.json",
        )

    results_df = pd.DataFrame(records)
    output_csv = METRICS_DIR / "multiview_cost_sensitive_comparison.csv"
    save_metrics_table(results_df.to_dict(orient="records"), output_csv)

    save_metrics_table_figure(
        results_df[
            [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "enrolled_recall",
                "enrolled_f1",
                "roc_auc_ovr_macro",
            ]
        ],
        FIGURE_DIR / "multiview_cost_sensitive_comparison.png",
        "Cost-Sensitive Multi-View Gated MLP",
    )

    save_confusion_matrix_grid(
        image_paths=cm_paths[:4],
        titles=cm_titles[:4],
        output_path=FIGURE_DIR / "multiview_cost_sensitive_confusion_grid.png",
        nrows=2,
        ncols=2,
    )

    print(results_df.to_string(index=False))
    print(f"saved comparison csv: {output_csv}")
    print(f"saved comparison figure: {FIGURE_DIR / 'multiview_cost_sensitive_comparison.png'}")


if __name__ == "__main__":
    main()
