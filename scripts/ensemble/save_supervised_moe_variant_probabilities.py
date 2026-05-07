"""Save cached probabilities for two supervised MoE variants.

This is the one-time cache builder for direct weighted voting. The vote script
itself (`weighted_merge_supervised_moe_variants.py`) never trains models.

Saved file:
    models/metrics/supervised_moe_variant_probabilities.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import search_supervised_error_router_moe as base
import search_supervised_error_router_moe_v2 as v2

from src.config import METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import save_metrics_table
from src.preprocessing import split_feature_columns


DEFAULT_CACHE = METRICS_DIR / "supervised_moe_variant_probabilities.npz"


def build_moe_variant(weight_mode, train_df, test_df, oof_probas, test_probas, losses, y_router, cv_splits, n_jobs):
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])
    train_router, test_router = v2.make_router_features(X_train, X_test, oof_probas, test_probas, "raw")
    numeric_features, categorical_features = split_feature_columns(train_router.columns)
    sample_weight = v2.margin_weights(losses, y_router, weight_mode)

    _, router_oof_soft = v2.router_oof_predictions(
        "logistic_router",
        train_router,
        y_router,
        len(oof_probas),
        cv_splits,
        numeric_features,
        categorical_features,
        n_jobs,
        sample_weight,
    )
    router = v2.make_router("logistic_router", numeric_features, categorical_features, RANDOM_STATE, n_jobs)
    v2.fit_router(router, train_router, y_router, sample_weight)
    router_test_soft = base.align_router_proba(router.predict_proba(test_router), router.classes_, len(test_probas))

    oof = (np.stack(oof_probas, axis=1) * router_oof_soft[:, :, None]).sum(axis=1)
    test = (np.stack(test_probas, axis=1) * router_test_soft[:, :, None]).sum(axis=1)
    return oof, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_train = train_df[TARGET_COL].astype(str).to_numpy()
    y_test = test_df[TARGET_COL].astype(str).to_numpy()
    numeric_features, categorical_features = split_feature_columns(train_df.drop(columns=[TARGET_COL]).columns)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    member_names, member_oof, member_test, member_metrics = base.build_member_probabilities(
        train_df,
        test_df,
        cv,
        numeric_features,
        categorical_features,
        args.n_jobs,
        quick=True,
    )
    losses = base.true_class_losses(member_oof, y_train)
    y_router = base.pseudo_labels_from_losses(losses, "raw_loss")

    best_oof, best_test = build_moe_variant(
        "none",
        train_df,
        test_df,
        member_oof,
        member_test,
        losses,
        y_router,
        args.cv,
        args.n_jobs,
    )
    enrolled_oof, enrolled_test = build_moe_variant(
        "sqrt_margin",
        train_df,
        test_df,
        member_oof,
        member_test,
        losses,
        y_router,
        args.cv,
        args.n_jobs,
    )

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.cache,
        y_train=y_train,
        y_test=y_test,
        best_oof=best_oof,
        best_test=best_test,
        enrolled_oof=enrolled_oof,
        enrolled_test=enrolled_test,
        member_names=np.array(member_names, dtype=object),
        labels=np.array(base.LABELS, dtype=object),
    )

    records = list(member_metrics)
    for name, oof, test in [
        ("moe_best_macro_raw_logistic_none_soft", best_oof, best_test),
        ("moe_enrolled_raw_logistic_sqrt_margin_soft", enrolled_oof, enrolled_test),
    ]:
        row = base.evaluate(f"{name}_oof", y_train, oof)
        row["split"] = "oof"
        records.append(row)
        row = base.evaluate(f"{name}_test", y_test, test)
        row["split"] = "test"
        records.append(row)

    metrics_path = METRICS_DIR / "supervised_moe_variant_probability_cache_metrics.csv"
    save_metrics_table(records, metrics_path)
    meta_path = METRICS_DIR / "supervised_moe_variant_probabilities_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "cv": args.cv,
                "n_jobs": args.n_jobs,
                "members": member_names,
                "variants": {
                    "best_oof/best_test": "moe_best_macro_raw_logistic_none_soft",
                    "enrolled_oof/enrolled_test": "moe_enrolled_raw_logistic_sqrt_margin_soft",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(pd.DataFrame(records)[["model", "accuracy", "f1_macro", "roc_auc_ovr_macro", "enrolled_accuracy"]].tail(5).to_string(index=False))
    print(f"saved cache: {args.cache}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
