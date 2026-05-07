"""Optimize class-specific decision bias for the OOF probability ensemble.

Declaration:
- This script does not tune on the final test set.
- It loads the cached OOF/test probabilities from oof_probability_ensemble.py.
- It searches class-specific log-probability biases on OOF predictions to
  maximize macro-F1.
- It then applies the selected OOF bias once to the held-out test predictions.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, LABELS, METRICS_DIR
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_metrics_table_figure


BIAS_VALUES = np.round(np.arange(-0.80, 0.85, 0.05), 2)
EPS = 1e-12


def load_probability_cache():
    cache_override = os.environ.get("OOF_ENSEMBLE_PROBA_CACHE")
    path = Path(cache_override) if cache_override else METRICS_DIR / "oof_probability_ensemble_probas.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing probability cache: {path}. "
            "Run `python scripts/oof_probability_ensemble.py` first."
        )
    data = np.load(path, allow_pickle=True)
    return path, data


def apply_bias(proba, bias):
    logits = np.log(np.clip(proba, EPS, 1.0)) + bias.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    biased_proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred_idx = biased_proba.argmax(axis=1)
    pred_labels = np.array([LABELS[idx] for idx in pred_idx])
    return pred_labels, biased_proba


def evaluate(name, y_true, y_pred, y_prob=None):
    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, labels=LABELS if y_prob is not None else None)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["model"] = name
    for label in LABELS:
        metrics[f"{label.lower()}_precision"] = report[label]["precision"]
        metrics[f"{label.lower()}_recall"] = report[label]["recall"]
        metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]
    return metrics


def search_bias(y_true, oof_proba):
    records = []
    best_score = -np.inf
    best_bias = None

    # Dropout bias is fixed at zero because adding the same constant to all
    # classes leaves argmax decisions unchanged.
    for enrolled_bias in BIAS_VALUES:
        for graduate_bias in BIAS_VALUES:
            bias = np.array([0.0, enrolled_bias, graduate_bias], dtype=float)
            pred, _ = apply_bias(oof_proba, bias)
            score = f1_score(y_true, pred, average="macro", zero_division=0)
            records.append(
                {
                    "dropout_bias": 0.0,
                    "enrolled_bias": float(enrolled_bias),
                    "graduate_bias": float(graduate_bias),
                    "oof_f1_macro": float(score),
                }
            )
            if score > best_score:
                best_score = score
                best_bias = bias

    grid_df = pd.DataFrame(records).sort_values("oof_f1_macro", ascending=False)
    return best_bias, best_score, grid_df


def main():
    cache_path, data = load_probability_cache()
    output_prefix = os.environ.get("OOF_ENSEMBLE_BIAS_PREFIX", "oof_ensemble_decision_bias")
    y_train = data["y_train"].astype(str)
    y_test = data["y_test"].astype(str)
    ensemble_oof = data["ensemble_oof"]
    ensemble_test = data["ensemble_test"]

    raw_oof_pred = np.array([LABELS[idx] for idx in ensemble_oof.argmax(axis=1)])
    raw_test_pred = np.array([LABELS[idx] for idx in ensemble_test.argmax(axis=1)])
    raw_oof_metrics = evaluate("oof_probability_ensemble_raw_oof", y_train, raw_oof_pred, y_prob=ensemble_oof)
    raw_test_metrics = evaluate("oof_probability_ensemble_raw_test", y_test, raw_test_pred, y_prob=ensemble_test)

    best_bias, best_score, grid_df = search_bias(y_train, ensemble_oof)
    biased_oof_pred, biased_oof_proba = apply_bias(ensemble_oof, best_bias)
    biased_test_pred, biased_test_proba = apply_bias(ensemble_test, best_bias)
    biased_oof_metrics = evaluate(
        "oof_probability_ensemble_biased_oof",
        y_train,
        biased_oof_pred,
        y_prob=biased_oof_proba,
    )
    biased_test_metrics = evaluate(
        "oof_probability_ensemble_biased_test",
        y_test,
        biased_test_pred,
        y_prob=biased_test_proba,
    )

    for model_name, pred in [
        ("oof_probability_ensemble_raw_test", raw_test_pred),
        ("oof_probability_ensemble_biased_test", biased_test_pred),
    ]:
        details = classification_details(y_test, pred, labels=LABELS)
        save_confusion_matrix(
            details["confusion_matrix"],
            LABELS,
            FIGURE_DIR / f"cm_{model_name}.png",
            f"Confusion Matrix: {model_name}",
        )

    metrics = [raw_oof_metrics, biased_oof_metrics, raw_test_metrics, biased_test_metrics]
    metrics_path = METRICS_DIR / f"{output_prefix}_metrics.csv"
    save_metrics_table(metrics, metrics_path)

    grid_path = METRICS_DIR / f"{output_prefix}_grid.csv"
    save_metrics_table(grid_df.to_dict(orient="records"), grid_path)

    payload = {
        "labels": LABELS,
        "bias": {label: float(value) for label, value in zip(LABELS, best_bias)},
        "oof_f1_macro": float(best_score),
        "source_probability_cache": str(cache_path),
        "selection_rule": "maximize OOF macro-F1; held-out test used only once for evaluation",
    }
    bias_path = METRICS_DIR / f"{output_prefix}_best.json"
    with open(bias_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    save_metrics_table_figure(
        pd.DataFrame([raw_test_metrics, biased_test_metrics])[
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
        FIGURE_DIR / "oof_ensemble_decision_bias_summary.png",
        "OOF Ensemble Decision Bias",
    )

    print(f"[PROGRESS] best_oof_bias={best_bias.tolist()} best_oof_f1_macro={best_score:.6f}", flush=True)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved bias grid: {grid_path}")
    print(f"saved best bias: {bias_path}")


if __name__ == "__main__":
    main()
