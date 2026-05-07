"""Switch to the enrolled-oriented MoE only when it predicts Enrolled.

This script does not train anything. It reads cached probabilities from:

    models/metrics/supervised_moe_variant_probabilities.npz

Rule:
    if enrolled-oriented MoE argmax == "Enrolled":
        use enrolled-oriented MoE probabilities
    else:
        use best-macro MoE probabilities
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


def switch_on_enrolled(best_proba, enrolled_proba):
    enrolled_idx = LABELS.index("Enrolled")
    use_enrolled = enrolled_proba.argmax(axis=1) == enrolled_idx
    merged = best_proba.copy()
    merged[use_enrolled] = enrolled_proba[use_enrolled]
    return merged, use_enrolled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-prefix", default="switch_supervised_moe_on_enrolled")
    args = parser.parse_args()

    if not args.cache.exists():
        raise FileNotFoundError(f"Missing cached probabilities: {args.cache}")

    payload = np.load(args.cache, allow_pickle=True)
    y_train = payload["y_train"].astype(str)
    y_test = payload["y_test"].astype(str)
    best_oof = payload["best_oof"]
    best_test = payload["best_test"]
    enrolled_oof = payload["enrolled_oof"]
    enrolled_test = payload["enrolled_test"]

    switched_oof, use_enrolled_oof = switch_on_enrolled(best_oof, enrolled_oof)
    switched_test, use_enrolled_test = switch_on_enrolled(best_test, enrolled_test)

    records = []
    for name, split, y_true, proba in [
        ("moe_best_macro_raw_logistic_none_soft", "oof", y_train, best_oof),
        ("moe_best_macro_raw_logistic_none_soft", "test", y_test, best_test),
        ("moe_enrolled_raw_logistic_sqrt_margin_soft", "oof", y_train, enrolled_oof),
        ("moe_enrolled_raw_logistic_sqrt_margin_soft", "test", y_test, enrolled_test),
        ("moe_switch_to_enrolled_when_second_predicts_enrolled", "oof", y_train, switched_oof),
        ("moe_switch_to_enrolled_when_second_predicts_enrolled", "test", y_test, switched_test),
    ]:
        row = base.evaluate(f"{name}_{split}", y_true, proba)
        row["split"] = split
        if name.startswith("moe_switch"):
            mask = use_enrolled_oof if split == "oof" else use_enrolled_test
            row["switch_count"] = int(mask.sum())
            row["switch_share"] = float(mask.mean())
        records.append(row)

    metrics_df = pd.DataFrame(records)
    best = metrics_df[metrics_df["split"] == "test"].sort_values("f1_macro", ascending=False).iloc[0].to_dict()
    switch_payload = {
        "rule": "Use enrolled-oriented MoE probabilities iff its argmax class is Enrolled; otherwise use best-macro MoE.",
        "cache": str(args.cache),
        "oof_switch_count": int(use_enrolled_oof.sum()),
        "oof_switch_share": float(use_enrolled_oof.mean()),
        "test_switch_count": int(use_enrolled_test.sum()),
        "test_switch_share": float(use_enrolled_test.mean()),
    }

    metrics_path = METRICS_DIR / f"{args.output_prefix}_metrics.csv"
    best_path = METRICS_DIR / f"{args.output_prefix}_best.json"
    rule_path = METRICS_DIR / f"{args.output_prefix}_rule.json"
    save_metrics_table(metrics_df.to_dict(orient="records"), metrics_path)
    best_path.write_text(json.dumps(json_safe(best), ensure_ascii=False, indent=2), encoding="utf-8")
    rule_path.write_text(json.dumps(json_safe(switch_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    print(metrics_df[metrics_df["split"] == "test"][[
        "model",
        "accuracy",
        "f1_macro",
        "roc_auc_ovr_macro",
        "fmi",
        "dropout_accuracy",
        "enrolled_accuracy",
        "graduate_accuracy",
        "switch_count",
        "switch_share",
    ]].sort_values("f1_macro", ascending=False).to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved rule: {rule_path}")


if __name__ == "__main__":
    main()
