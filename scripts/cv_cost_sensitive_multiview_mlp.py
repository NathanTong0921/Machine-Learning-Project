"""Run a lightweight 5-fold CV for cost-sensitive Multi-View Gated MLP variants.

Declaration:
- This script evaluates cost-sensitive loss variants on the processed training set only.
- The held-out processed test set is not used here.
- Each outer fold still uses the model's internal validation split for early stopping.
"""

import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import train_multiview_gated_mlp
from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table


BASE_CONFIG = {
    "seed": 62,
    "hidden_dim": 64,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 60,
    "patience": 8,
}

VARIANTS = [
    ("baseline_recheck", {}),
    ("enrolled_x1_15", {"Enrolled": 1.15}),
    ("enrolled_x1_30_dropout_015", {"Enrolled": 1.30}, {"dropout": 0.15}),
]


def evaluate_fold(model_name, result):
    metrics = classification_metrics(
        result["test_true_labels"],
        result["test_pred_labels"],
        y_prob=result["test_prob"],
        labels=LABELS,
    )
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
    return metrics


def main():
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("Missing processed train CSV. Run `python scripts/prepare_data.py` first.")

    full_df = pd.read_csv(train_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_records = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(full_df, full_df[TARGET_COL]), start=1):
        fold_train_df = full_df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = full_df.iloc[valid_idx].reset_index(drop=True)

        for variant in VARIANTS:
            name = variant[0]
            class_weight_multipliers = variant[1]
            config_overrides = variant[2] if len(variant) > 2 else {}
            config = {**BASE_CONFIG, **config_overrides, "seed": BASE_CONFIG["seed"] + fold_idx}

            print(
                f"[PROGRESS] fold={fold_idx}/5 variant={name} "
                f"class_weight_multipliers={class_weight_multipliers}",
                flush=True,
            )
            result = train_multiview_gated_mlp(
                fold_train_df,
                fold_valid_df,
                class_weight_multipliers=class_weight_multipliers,
                **config,
            )
            metrics = evaluate_fold(f"multiview_cost_sensitive_{name}", result)
            metrics["fold"] = fold_idx
            metrics["variant"] = name
            metrics["class_weight_multipliers"] = json.dumps(class_weight_multipliers, sort_keys=True)
            metrics["best_epoch"] = result["metadata"]["best_epoch"]
            metrics["best_val_loss"] = result["metadata"]["best_val_loss"]
            fold_records.append(metrics)

    fold_df = pd.DataFrame(fold_records)
    fold_output = METRICS_DIR / "multiview_cost_sensitive_5fold.csv"
    save_metrics_table(fold_df.to_dict(orient="records"), fold_output)

    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
        "enrolled_recall",
        "enrolled_f1",
    ]
    summary_df = (
        fold_df.groupby(["model", "variant", "class_weight_multipliers"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary_df.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary_df.columns.to_flat_index()
    ]
    summary_output = METRICS_DIR / "multiview_cost_sensitive_5fold_summary.csv"
    save_metrics_table(summary_df.to_dict(orient="records"), summary_output)

    print(summary_df.to_string(index=False))
    print(f"saved fold csv: {fold_output}")
    print(f"saved summary csv: {summary_output}")


if __name__ == "__main__":
    main()
