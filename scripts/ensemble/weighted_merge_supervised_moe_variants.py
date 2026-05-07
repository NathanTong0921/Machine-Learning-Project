"""Direct weighted vote for cached supervised MoE variant probabilities.

This script does not train experts or routers. It only reads cached OOF/test
probabilities for two standalone MoE variants and evaluates direct probability
averaging.

Expected cache:
    models/metrics/supervised_moe_variant_probabilities.npz

Required arrays:
    y_train, y_test
    best_oof, best_test
    enrolled_oof, enrolled_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import METRICS_DIR
from src.evaluation import save_metrics_table

import search_supervised_error_router_moe as base


DEFAULT_CACHE = METRICS_DIR / "supervised_moe_variant_probabilities.npz"
VARIANTS = {
    "best_macro": "moe_best_macro_raw_logistic_none_soft",
    "enrolled": "moe_enrolled_raw_logistic_sqrt_margin_soft",
}


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


def weighted_vote(weight_best, best_proba, enrolled_proba):
    return weight_best * best_proba + (1.0 - weight_best) * enrolled_proba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output-prefix", default="weighted_merge_supervised_moe_variants")
    args = parser.parse_args()

    if not args.cache.exists():
        raise FileNotFoundError(
            f"Missing cached MoE probabilities: {args.cache}. "
            "This vote script intentionally does not retrain models. Save "
            "best_oof/best_test/enrolled_oof/enrolled_test first."
        )

    payload = np.load(args.cache, allow_pickle=True)
    y_train = payload["y_train"].astype(str)
    y_test = payload["y_test"].astype(str)
    best_oof = payload["best_oof"]
    best_test = payload["best_test"]
    enrolled_oof = payload["enrolled_oof"]
    enrolled_test = payload["enrolled_test"]

    records = []
    for key, name in VARIANTS.items():
        oof = best_oof if key == "best_macro" else enrolled_oof
        test = best_test if key == "best_macro" else enrolled_test
        row = base.evaluate(f"{name}_oof", y_train, oof)
        row.update({"split": "oof", "weight_best_macro": 1.0 if key == "best_macro" else 0.0})
        records.append(row)
        row = base.evaluate(f"{name}_test", y_test, test)
        row.update({"split": "test", "weight_best_macro": 1.0 if key == "best_macro" else 0.0})
        records.append(row)

    grid = []
    n_steps = int(round(1.0 / args.step))
    for i in range(n_steps + 1):
        weight_best = i / n_steps
        oof_vote = weighted_vote(weight_best, best_oof, enrolled_oof)
        oof_metrics = base.evaluate("tmp", y_train, oof_vote)
        grid.append({
            "weight_best_macro": weight_best,
            "weight_enrolled": 1.0 - weight_best,
            "oof_accuracy": oof_metrics["accuracy"],
            "oof_f1_macro": oof_metrics["f1_macro"],
            "oof_enrolled_accuracy": oof_metrics["enrolled_accuracy"],
        })

    grid_df = pd.DataFrame(grid).sort_values("oof_f1_macro", ascending=False)
    best_weight = float(grid_df.iloc[0]["weight_best_macro"])

    for name, weight_best in [
        ("moe_equal_soft_vote", 0.5),
        ("moe_oof_weighted_soft_vote", best_weight),
    ]:
        for split, y_true, best_proba, enrolled_proba in [
            ("oof", y_train, best_oof, enrolled_oof),
            ("test", y_test, best_test, enrolled_test),
        ]:
            proba = weighted_vote(weight_best, best_proba, enrolled_proba)
            row = base.evaluate(f"{name}_{split}", y_true, proba)
            row.update({
                "split": split,
                "weight_best_macro": weight_best,
                "weight_enrolled": 1.0 - weight_best,
                "weights_json": json.dumps(
                    {
                        VARIANTS["best_macro"]: weight_best,
                        VARIANTS["enrolled"]: 1.0 - weight_best,
                    },
                    sort_keys=True,
                ),
            })
            records.append(row)

    metrics_df = pd.DataFrame(records)
    test_rank = metrics_df[metrics_df["split"] == "test"].sort_values("f1_macro", ascending=False)
    best = test_rank.iloc[0].to_dict()
    weights_payload = {
        "selection": "OOF macro-F1",
        "cache": str(args.cache),
        "variants": VARIANTS,
        "best_weight_by_oof": {
            VARIANTS["best_macro"]: best_weight,
            VARIANTS["enrolled"]: 1.0 - best_weight,
        },
        "equal_weight": {
            VARIANTS["best_macro"]: 0.5,
            VARIANTS["enrolled"]: 0.5,
        },
    }

    metrics_path = METRICS_DIR / f"{args.output_prefix}_metrics.csv"
    grid_path = METRICS_DIR / f"{args.output_prefix}_weight_grid.csv"
    best_path = METRICS_DIR / f"{args.output_prefix}_best.json"
    weights_path = METRICS_DIR / f"{args.output_prefix}_weights.json"
    save_metrics_table(metrics_df.to_dict(orient="records"), metrics_path)
    save_metrics_table(grid_df.to_dict(orient="records"), grid_path)
    best_path.write_text(json.dumps(json_safe(best), ensure_ascii=False, indent=2), encoding="utf-8")
    weights_path.write_text(json.dumps(json_safe(weights_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    print(test_rank[[
        "model",
        "accuracy",
        "f1_macro",
        "roc_auc_ovr_macro",
        "fmi",
        "dropout_accuracy",
        "enrolled_accuracy",
        "graduate_accuracy",
    ]].to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved weights: {weights_path}")


if __name__ == "__main__":
    main()
