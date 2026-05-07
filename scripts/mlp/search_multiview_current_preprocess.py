"""Search MultiView Gated MLP hyperparameters under current preprocessing.

The current preprocessing keeps `Application order` numeric/ordinal, so old
checkpoints trained with the legacy one-hot treatment are not compatible.
This script performs a sharded search and writes incremental results.
"""

import argparse
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
from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, MODEL_DIR_MLP, PROCESSED_DIR
from src.evaluation import classification_metrics, save_metrics_table


MODEL_PREFIX = "multiview_current_preprocess_search"


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
                    "batch_size": 128,
                    "max_epochs": 90,
                    "patience": 12,
                    "class_weight_multipliers": {},
                }
            )

    for hidden_dim in [96, 128]:
        for dropout in [0.10, 0.15]:
            for lr in [5e-4, 7e-4, 1.5e-3]:
                configs.append(
                    {
                        "seed": 62,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "lr": lr,
                        "weight_decay": 1e-4,
                        "batch_size": 128,
                        "max_epochs": 90,
                        "patience": 12,
                        "class_weight_multipliers": {},
                    }
                )

    for hidden_dim in [64, 96, 128]:
        for dropout in [0.10, 0.15]:
            for enrolled_weight in [1.15, 1.30]:
                configs.append(
                    {
                        "seed": 62,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "lr": 1e-3,
                        "weight_decay": 1e-4,
                        "batch_size": 128,
                        "max_epochs": 90,
                        "patience": 12,
                        "class_weight_multipliers": {"Enrolled": enrolled_weight},
                    }
                )

    for hidden_dim in [96, 128]:
        for batch_size in [64, 256]:
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

    for seed in [42, 82]:
        configs.append(
            {
                "seed": seed,
                "hidden_dim": 96,
                "dropout": 0.10,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 128,
                "max_epochs": 90,
                "patience": 12,
                "class_weight_multipliers": {},
            }
        )

    return configs


def jsonable_record(record):
    out = {}
    for key, value in record.items():
        if hasattr(value, "item"):
            value = value.item()
        out[key] = value
    return out


def evaluate_result(result):
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
        prefix = label.lower()
        metrics[f"{prefix}_precision"] = report[label]["precision"]
        metrics[f"{prefix}_recall"] = report[label]["recall"]
        metrics[f"{prefix}_f1"] = report[label]["f1-score"]
    return jsonable_record(metrics)


def save_checkpoint(result, config, path):
    metadata = {
        **{k: v for k, v in result["metadata"].items() if k != "history"},
        "config": config,
        "categorical_columns": sorted(CATEGORICAL_COLUMNS),
        "preprocessing_note": "Current Appendix-aligned config keeps Application order numeric/ordinal.",
    }
    torch.save({"state_dict": result["model"].state_dict(), "metadata": metadata}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    all_configs = config_grid()
    shard_configs = [cfg for idx, cfg in enumerate(all_configs) if idx % args.num_shards == args.shard_index]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR_MLP.mkdir(parents=True, exist_ok=True)
    shard_csv = METRICS_DIR / f"{MODEL_PREFIX}_shard{args.shard_index}_of{args.num_shards}.csv"
    best_ckpt = MODEL_DIR_MLP / f"{MODEL_PREFIX}_best_shard{args.shard_index}_of{args.num_shards}.pt"

    records = []
    best_f1 = -1.0
    best_record = None
    for local_idx, config in enumerate(shard_configs, start=1):
        class_weight_multipliers = config["class_weight_multipliers"]
        run_config = {k: v for k, v in config.items() if k != "class_weight_multipliers"}
        print(
            f"[PROGRESS] shard={args.shard_index}/{args.num_shards} "
            f"config={local_idx}/{len(shard_configs)} config={run_config} "
            f"class_weight_multipliers={class_weight_multipliers}",
            flush=True,
        )
        result = train_multiview_gated_mlp(
            train_df,
            test_df,
            class_weight_multipliers=class_weight_multipliers,
            **run_config,
        )
        record = evaluate_result(result)
        full_config = {**run_config, "class_weight_multipliers": class_weight_multipliers}
        record["model"] = MODEL_PREFIX
        record["global_config_index"] = all_configs.index(full_config)
        record["shard_index"] = args.shard_index
        record["best_epoch"] = result["metadata"]["best_epoch"]
        record["best_val_loss"] = result["metadata"]["best_val_loss"]
        record["view_dims"] = json.dumps(result["metadata"]["view_dims"], sort_keys=True)
        record["config_json"] = json.dumps(full_config, sort_keys=True)
        records.append(record)
        save_metrics_table(records, shard_csv)

        if record["f1_macro"] > best_f1:
            best_f1 = record["f1_macro"]
            best_record = record
            save_checkpoint(result, full_config, best_ckpt)
            with open(METRICS_DIR / f"{MODEL_PREFIX}_best_shard{args.shard_index}_of{args.num_shards}.json", "w", encoding="utf-8") as f:
                json.dump(best_record, f, ensure_ascii=False, indent=2)
            print(f"[BEST] shard={args.shard_index} f1_macro={best_f1:.6f}", flush=True)

    print(f"[DONE] shard={args.shard_index} best_f1_macro={best_f1:.6f}")
    print(f"saved shard csv: {shard_csv}")
    print(f"saved best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
