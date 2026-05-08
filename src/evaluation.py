"""Classification and clustering evaluation helpers."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fowlkes_mallows_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(y_true, y_pred, y_prob=None, labels=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "fmi": fowlkes_mallows_score(y_true, y_pred),
    }

    if labels is not None:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        support = cm.sum(axis=1)
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        for idx, label in enumerate(labels):
            key = str(label).lower()
            metrics[f"{key}_accuracy"] = cm[idx, idx] / support[idx] if support[idx] else 0.0
            metrics[f"{key}_precision"] = per_class_precision[idx]
            metrics[f"{key}_recall"] = per_class_recall[idx]
            metrics[f"{key}_f1"] = per_class_f1[idx]

    if y_prob is not None and labels is not None:
        y_true_bin = label_binarize(y_true, classes=labels)
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_true_bin,
            y_prob,
            average="macro",
            multi_class="ovr",
        )
    return metrics


def classification_details(y_true, y_pred, labels):
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
    }


def save_metrics_table(records, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
