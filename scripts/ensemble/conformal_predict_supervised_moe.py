"""Conformal prediction sets for the supervised Core3 MoE.

This script does not retrain the base model. It reads cached OOF/test
probabilities from:

    models/metrics/supervised_moe_variant_probabilities.npz

Two conformal scores are evaluated:

- THR: threshold conformal score, 1 - p_true.
- APS: adaptive prediction set score, cumulative sorted probability up to the
  true class.

The OOF train probabilities are used as calibration scores. This is a
cross-conformal-style diagnostic rather than a clean split-conformal guarantee;
for a strict formal guarantee, reserve a standalone calibration split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import search_supervised_error_router_moe as base

from src.config import LABELS, METRICS_DIR
from src.evaluation import save_metrics_table


DEFAULT_CACHE = METRICS_DIR / "supervised_moe_variant_probabilities.npz"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return value


def conformal_quantile(scores, alpha):
    n = len(scores)
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def true_class_indices(y):
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    return np.array([label_to_idx[str(label)] for label in y], dtype=np.int64)


def thr_scores(proba, y):
    idx = true_class_indices(y)
    return 1.0 - proba[np.arange(len(y)), idx]


def aps_score_matrix(proba):
    order = np.argsort(-proba, axis=1)
    sorted_prob = np.take_along_axis(proba, order, axis=1)
    cum_prob = np.cumsum(sorted_prob, axis=1)
    scores = np.zeros_like(proba)
    np.put_along_axis(scores, order, cum_prob, axis=1)
    return scores


def aps_scores(proba, y):
    idx = true_class_indices(y)
    score_matrix = aps_score_matrix(proba)
    return score_matrix[np.arange(len(y)), idx]


def prediction_sets(method, proba, qhat):
    if method == "thr":
        return (1.0 - proba) <= qhat
    if method == "aps":
        return aps_score_matrix(proba) <= qhat
    raise ValueError(f"Unknown method: {method}")


def set_to_text(mask):
    labels = [label for label, keep in zip(LABELS, mask) if keep]
    return "{" + ",".join(labels) + "}"


def evaluate_sets(method, alpha, qhat, y_true, proba, masks, split):
    y_true = np.asarray(y_true, dtype=str)
    y_idx = true_class_indices(y_true)
    covered = masks[np.arange(len(y_true)), y_idx]
    set_size = masks.sum(axis=1)
    singleton = set_size == 1
    empty = set_size == 0
    pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
    hard_metrics = base.evaluate(f"hard_argmax_{split}", y_true, proba)

    rows = {
        "method": method,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "qhat": qhat,
        "split": split,
        "coverage": float(covered.mean()),
        "avg_set_size": float(set_size.mean()),
        "median_set_size": float(np.median(set_size)),
        "singleton_rate": float(singleton.mean()),
        "empty_rate": float(empty.mean()),
        "hard_accuracy": hard_metrics["accuracy"],
        "hard_f1_macro": hard_metrics["f1_macro"],
        "hard_auc": hard_metrics["roc_auc_ovr_macro"],
        "ambiguous_enrolled_graduate_rate": float(
            (masks[:, LABELS.index("Enrolled")] & masks[:, LABELS.index("Graduate")] & (set_size == 2)).mean()
        ),
        "contains_enrolled_rate": float(masks[:, LABELS.index("Enrolled")].mean()),
        "contains_graduate_rate": float(masks[:, LABELS.index("Graduate")].mean()),
        "contains_dropout_rate": float(masks[:, LABELS.index("Dropout")].mean()),
    }
    for label in LABELS:
        label_mask = y_true == label
        key = label.lower()
        rows[f"{key}_coverage"] = float(covered[label_mask].mean()) if label_mask.any() else np.nan
        rows[f"{key}_avg_set_size"] = float(set_size[label_mask].mean()) if label_mask.any() else np.nan
    return json_safe(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--variant", choices=["best", "enrolled"], default="best")
    parser.add_argument("--alphas", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--output-prefix", default="conformal_supervised_moe")
    args = parser.parse_args()

    if not args.cache.exists():
        raise FileNotFoundError(f"Missing cached MoE probabilities: {args.cache}")

    payload = np.load(args.cache, allow_pickle=True)
    y_train = payload["y_train"].astype(str)
    y_test = payload["y_test"].astype(str)
    if args.variant == "best":
        train_proba = payload["best_oof"]
        test_proba = payload["best_test"]
        variant_name = "moe_best_macro_raw_logistic_none_soft"
    else:
        train_proba = payload["enrolled_oof"]
        test_proba = payload["enrolled_test"]
        variant_name = "moe_enrolled_raw_logistic_sqrt_margin_soft"

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    score_builders = {
        "thr": thr_scores,
        "aps": aps_scores,
    }
    records = []
    prediction_rows = []
    for method, score_fn in score_builders.items():
        cal_scores = score_fn(train_proba, y_train)
        for alpha in alphas:
            qhat = conformal_quantile(cal_scores, alpha)
            for split, y_true, proba in [("oof_calibration", y_train, train_proba), ("test", y_test, test_proba)]:
                masks = prediction_sets(method, proba, qhat)
                records.append(evaluate_sets(method, alpha, qhat, y_true, proba, masks, split))
                if split == "test":
                    pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
                    for idx in range(len(y_true)):
                        prediction_rows.append(
                            {
                                "method": method,
                                "alpha": alpha,
                                "row_id": idx,
                                "true_label": str(y_true[idx]),
                                "hard_pred": str(pred[idx]),
                                "set_size": int(masks[idx].sum()),
                                "prediction_set": set_to_text(masks[idx]),
                                "p_dropout": float(proba[idx, LABELS.index("Dropout")]),
                                "p_enrolled": float(proba[idx, LABELS.index("Enrolled")]),
                                "p_graduate": float(proba[idx, LABELS.index("Graduate")]),
                            }
                        )

    metrics_df = pd.DataFrame(records)
    pred_df = pd.DataFrame(prediction_rows)
    metrics_path = METRICS_DIR / f"{args.output_prefix}_{args.variant}_metrics.csv"
    pred_path = METRICS_DIR / f"{args.output_prefix}_{args.variant}_test_sets.csv"
    config_path = METRICS_DIR / f"{args.output_prefix}_{args.variant}_config.json"
    save_metrics_table(metrics_df.to_dict(orient="records"), metrics_path)
    save_metrics_table(pred_df.to_dict(orient="records"), pred_path)
    config_path.write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "variant": variant_name,
                "labels": LABELS,
                "alphas": alphas,
                "methods": list(score_builders),
                "calibration_note": "OOF train probabilities are used as calibration scores; strict split conformal would require a reserved calibration split.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics_df[metrics_df["split"] == "test"][[
        "method",
        "alpha",
        "nominal_coverage",
        "coverage",
        "avg_set_size",
        "singleton_rate",
        "ambiguous_enrolled_graduate_rate",
        "dropout_coverage",
        "enrolled_coverage",
        "graduate_coverage",
    ]].to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved test sets: {pred_path}")


if __name__ == "__main__":
    main()
