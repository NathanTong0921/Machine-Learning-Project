"""Compare Application order encodings for MultiView Gated MLP.

The current Appendix-aligned preprocessing treats `Application order` as a
numeric/ordinal scalar. This script tests whether replacing that scalar with
thermometer or OOF target-encoding features helps the MultiView MLP.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import train_multiview_gated_mlp
from src.config import FEATURE_GROUPS, FIGURE_DIR, LABELS, METRICS_DIR, MODEL_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix


APP_ORDER_COL = "Application order"
MODEL_PREFIX = "multiview_application_order_encoding"

BASE_CONFIG = {
    "seed": 62,
    "hidden_dim": 96,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 90,
    "patience": 12,
}


def json_safe(record):
    out = {}
    for key, value in record.items():
        if hasattr(value, "item"):
            value = value.item()
        out[key] = value
    return out


def replace_application_order(feature_columns, new_columns):
    return [
        replacement
        for col in feature_columns
        for replacement in (new_columns if col == APP_ORDER_COL else [col])
    ]


def make_feature_groups(new_application_order_columns):
    groups = {name: list(cols) for name, cols in FEATURE_GROUPS.items()}
    groups["enrollment_profile"] = replace_application_order(
        groups["enrollment_profile"],
        new_application_order_columns,
    )
    return groups


def add_thermometer_features(train_df, test_df):
    train_out = train_df.copy()
    test_out = test_df.copy()
    columns = []
    for threshold in range(1, 10):
        col = f"Application order >= {threshold}"
        train_out[col] = (train_out[APP_ORDER_COL] >= threshold).astype(float)
        test_out[col] = (test_out[APP_ORDER_COL] >= threshold).astype(float)
        columns.append(col)
    return train_out, test_out, columns


def smoothed_class_rates(frame, labels, alpha):
    counts = frame[TARGET_COL].value_counts().reindex(labels, fill_value=0).astype(float)
    total = float(len(frame))
    return (counts + alpha) / (total + alpha * len(labels))


def add_oof_target_encoding(train_df, test_df, n_splits=5, alpha=10.0):
    train_out = train_df.copy()
    test_out = test_df.copy()
    columns = [f"Application order te {label.lower()}" for label in LABELS]
    for col in columns:
        train_out[col] = np.nan

    global_rates = smoothed_class_rates(train_df, LABELS, alpha)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=62)
    for train_idx, valid_idx in cv.split(train_df, train_df[TARGET_COL]):
        fold_train = train_df.iloc[train_idx]
        fold_global = smoothed_class_rates(fold_train, LABELS, alpha)
        grouped = fold_train.groupby(APP_ORDER_COL)
        maps = {}
        for order_value, group in grouped:
            rates = smoothed_class_rates(group, LABELS, alpha)
            weight = len(group) / (len(group) + alpha)
            maps[order_value] = weight * rates + (1.0 - weight) * fold_global

        for row_idx in valid_idx:
            order_value = train_df.iloc[row_idx][APP_ORDER_COL]
            rates = maps.get(order_value, fold_global)
            for label, col in zip(LABELS, columns):
                train_out.iat[row_idx, train_out.columns.get_loc(col)] = rates[label]

    full_maps = {}
    for order_value, group in train_df.groupby(APP_ORDER_COL):
        rates = smoothed_class_rates(group, LABELS, alpha)
        weight = len(group) / (len(group) + alpha)
        full_maps[order_value] = weight * rates + (1.0 - weight) * global_rates

    for col in columns:
        test_out[col] = np.nan
    for row_idx, order_value in enumerate(test_df[APP_ORDER_COL]):
        rates = full_maps.get(order_value, global_rates)
        for label, col in zip(LABELS, columns):
            test_out.iat[row_idx, test_out.columns.get_loc(col)] = rates[label]

    train_out[columns] = train_out[columns].astype(float)
    test_out[columns] = test_out[columns].astype(float)
    return train_out, test_out, columns


def evaluate_variant(name, result, feature_groups, encoding_note):
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
    for label in LABELS:
        key = label.lower()
        metrics[f"{key}_precision"] = report[label]["precision"]
        metrics[f"{key}_recall"] = report[label]["recall"]
        metrics[f"{key}_f1"] = report[label]["f1-score"]
    metrics = json_safe(metrics)
    metrics["model"] = name
    metrics["encoding_note"] = encoding_note
    metrics["view_dims"] = json.dumps(result["metadata"]["view_dims"], sort_keys=True)
    metrics["config_json"] = json.dumps(BASE_CONFIG, sort_keys=True)
    metrics["feature_groups_json"] = json.dumps(feature_groups, sort_keys=True)
    metrics["best_epoch"] = result["metadata"]["best_epoch"]
    metrics["best_val_loss"] = result["metadata"]["best_val_loss"]
    return metrics


def save_artifacts(name, result, metrics):
    safe_name = f"{MODEL_PREFIX}_{name}"
    details = classification_details(result["test_true_labels"], result["test_pred_labels"], labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / f"cm_{safe_name}.png",
        f"Confusion Matrix: {safe_name}",
    )
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "metadata": {
                **{k: v for k, v in result["metadata"].items() if k != "history"},
                "metrics": metrics,
            },
        },
        MODEL_DIR / f"{safe_name}.pt",
    )


def run_variant(name, train_df, test_df, feature_groups, encoding_note):
    print(f"[RUN] {name}: {encoding_note}", flush=True)
    result = train_multiview_gated_mlp(
        train_df,
        test_df,
        feature_groups=feature_groups,
        **BASE_CONFIG,
    )
    metrics = evaluate_variant(name, result, feature_groups, encoding_note)
    save_artifacts(name, result, metrics)
    print(pd.DataFrame([metrics])[["model", "accuracy", "f1_macro", "roc_auc_ovr_macro", "enrolled_accuracy"]].to_string(index=False), flush=True)
    return metrics


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    records.append(
        run_variant(
            "scalar_current_best_recheck",
            train_df,
            test_df,
            FEATURE_GROUPS,
            "Current Appendix-aligned scalar numeric Application order.",
        )
    )

    thermo_train, thermo_test, thermo_cols = add_thermometer_features(train_df, test_df)
    records.append(
        run_variant(
            "thermometer_replace",
            thermo_train,
            thermo_test,
            make_feature_groups(thermo_cols),
            "Replace scalar Application order with nine cumulative threshold indicators.",
        )
    )

    te_train, te_test, te_cols = add_oof_target_encoding(train_df, test_df)
    records.append(
        run_variant(
            "target_encoding_oof_replace",
            te_train,
            te_test,
            make_feature_groups(te_cols),
            "Replace scalar Application order with 5-fold OOF smoothed class-rate target encoding.",
        )
    )

    out = METRICS_DIR / f"{MODEL_PREFIX}_metrics.csv"
    save_metrics_table(records, out)
    best = max(records, key=lambda row: row["f1_macro"])
    with open(METRICS_DIR / f"{MODEL_PREFIX}_best.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(pd.DataFrame(records).sort_values("f1_macro", ascending=False).to_string(index=False))
    print(f"saved metrics: {out}")


if __name__ == "__main__":
    main()
