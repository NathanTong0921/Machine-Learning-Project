"""Tune MultiView MLP for alternative Application order encodings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mlp.compare_application_order_encodings_multiview import (
    add_oof_target_encoding,
    add_thermometer_features,
    evaluate_variant,
    make_feature_groups,
    save_artifacts,
)
from src.config import FEATURE_GROUPS, METRICS_DIR, MODEL_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import save_metrics_table


MODEL_PREFIX = "multiview_application_order_encoding_search"


def config_grid():
    configs = []
    for hidden_dim in [64, 96, 128, 160]:
        for dropout in [0.05, 0.10, 0.15, 0.20]:
            configs.append(
                {
                    "seed": 62,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": 64,
                    "max_epochs": 90,
                    "patience": 12,
                    "class_weight_multipliers": {},
                }
            )

    for hidden_dim in [96, 128, 160]:
        for dropout in [0.10, 0.15]:
            for lr in [5e-4, 7e-4, 1.5e-3]:
                configs.append(
                    {
                        "seed": 62,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "lr": lr,
                        "weight_decay": 1e-4,
                        "batch_size": 64,
                        "max_epochs": 90,
                        "patience": 12,
                        "class_weight_multipliers": {},
                    }
                )

    for hidden_dim in [96, 128, 160]:
        for dropout in [0.10, 0.15]:
            for enrolled_weight in [1.10, 1.20, 1.30]:
                configs.append(
                    {
                        "seed": 62,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "lr": 1e-3,
                        "weight_decay": 1e-4,
                        "batch_size": 64,
                        "max_epochs": 90,
                        "patience": 12,
                        "class_weight_multipliers": {"Enrolled": enrolled_weight},
                    }
                )

    for hidden_dim in [96, 128, 160]:
        for batch_size in [128, 256]:
            configs.append(
                {
                    "seed": 62,
                    "hidden_dim": hidden_dim,
                    "dropout": 0.10,
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "batch_size": batch_size,
                    "max_epochs": 90,
                    "patience": 12,
                    "class_weight_multipliers": {},
                }
            )
    return configs


def build_variant_data(variant, train_df, test_df):
    if variant == "scalar":
        return train_df, test_df, FEATURE_GROUPS, "Current scalar numeric Application order."
    if variant == "thermometer":
        train_variant, test_variant, columns = add_thermometer_features(train_df, test_df)
        return (
            train_variant,
            test_variant,
            make_feature_groups(columns),
            "Replace scalar Application order with nine cumulative threshold indicators.",
        )
    if variant == "target_encoding_oof":
        train_variant, test_variant, columns = add_oof_target_encoding(train_df, test_df)
        return (
            train_variant,
            test_variant,
            make_feature_groups(columns),
            "Replace scalar Application order with 5-fold OOF smoothed class-rate target encoding.",
        )
    raise ValueError(f"Unknown variant: {variant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["scalar", "thermometer", "target_encoding_oof"], required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    train_variant, test_variant, feature_groups, note = build_variant_data(args.variant, train_df, test_df)

    records = []
    best_record = None
    best_checkpoint = None
    variant_prefix = f"{MODEL_PREFIX}_{args.variant}"
    csv_path = METRICS_DIR / f"{variant_prefix}.csv"
    best_json_path = METRICS_DIR / f"{variant_prefix}_best.json"

    for idx, config in enumerate(config_grid(), start=1):
        class_weight_multipliers = config["class_weight_multipliers"]
        run_config = {k: v for k, v in config.items() if k != "class_weight_multipliers"}
        print(f"[SEARCH] variant={args.variant} config={idx}/{len(config_grid())} config={config}", flush=True)

        from src.advanced_models import train_multiview_gated_mlp

        result = train_multiview_gated_mlp(
            train_variant,
            test_variant,
            feature_groups=feature_groups,
            class_weight_multipliers=class_weight_multipliers,
            **run_config,
        )
        record = evaluate_variant(
            f"{args.variant}_tuned",
            result,
            feature_groups,
            note,
        )
        full_config = {**run_config, "class_weight_multipliers": class_weight_multipliers}
        record["config_json"] = json.dumps(full_config, sort_keys=True)
        record["search_index"] = idx
        records.append(record)
        save_metrics_table(records, csv_path)

        if best_record is None or record["f1_macro"] > best_record["f1_macro"]:
            best_record = record
            best_checkpoint = MODEL_DIR / f"{variant_prefix}_best.pt"
            save_artifacts(f"{variant_prefix}_best", result, record)
            tmp_ckpt = MODEL_DIR / f"multiview_application_order_encoding_{variant_prefix}_best.pt"
            if tmp_ckpt.exists() and tmp_ckpt != best_checkpoint:
                shutil.copyfile(tmp_ckpt, best_checkpoint)
            with open(best_json_path, "w", encoding="utf-8") as f:
                json.dump(best_record, f, ensure_ascii=False, indent=2)
            print(f"[BEST] variant={args.variant} f1_macro={best_record['f1_macro']:.6f}", flush=True)

    df = pd.DataFrame(records).sort_values("f1_macro", ascending=False)
    df.to_csv(csv_path, index=False)
    print(df.head(10).to_string(index=False))
    print(f"saved csv: {csv_path}")
    print(f"saved best json: {best_json_path}")
    if best_checkpoint is not None:
        print(f"saved best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
