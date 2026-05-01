"""Run a small seed-stability check for the best FT-Transformer configuration.

Declaration:
- This script measures whether the tuned FT-Transformer is stable across seeds.
- It uses the current best FT configuration found in the tuning stage.

Tips:
- Read mean/std of macro-F1 before reading the best single run.
- If variance is large, do not over-interpret any one FT result.
"""

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import train_ft_transformer
from src.config import FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR
from src.evaluation import classification_metrics, save_metrics_table
from src.visualization import save_metrics_table_figure


BEST_FT_CONFIG = {
    "d_token": 48,
    "n_blocks": 2,
    "attention_n_heads": 4,
    "attention_dropout": 0.20,
    "ff_dropout": 0.20,
    "lr": 8e-4,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "max_epochs": 60,
    "patience": 10,
}

SEEDS = [42, 62, 82]


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    records = []
    best_record = None

    for seed in SEEDS:
        result = train_ft_transformer(
            train_df=train_df,
            test_df=test_df,
            seed=seed,
            **BEST_FT_CONFIG,
        )
        metrics = classification_metrics(
            result["test_true_labels"],
            result["test_pred_labels"],
            y_prob=result["test_prob"],
            labels=LABELS,
        )
        record = {
            "seed": seed,
            **metrics,
        }
        records.append(record)
        if best_record is None or record["f1_macro"] > best_record["f1_macro"]:
            best_record = record

    df = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)

    summary = {
        "config": BEST_FT_CONFIG,
        "seeds": SEEDS,
        "mean_accuracy": float(df["accuracy"].mean()),
        "std_accuracy": float(df["accuracy"].std(ddof=0)),
        "mean_f1_macro": float(df["f1_macro"].mean()),
        "std_f1_macro": float(df["f1_macro"].std(ddof=0)),
        "mean_auc": float(df["roc_auc_ovr_macro"].mean()),
        "std_auc": float(df["roc_auc_ovr_macro"].std(ddof=0)),
        "best_record": best_record,
    }

    output_csv = METRICS_DIR / "ft_seed_stability.csv"
    output_json = METRICS_DIR / "ft_seed_stability_summary.json"
    output_md = METRICS_DIR / "ft_seed_stability.md"
    output_png = FIGURE_DIR / "ft_seed_stability_table.png"

    save_metrics_table(df.to_dict(orient="records"), output_csv)
    save_metrics_table_figure(
        df[
            [
                "seed",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
            ]
        ],
        output_png,
        "FT-Transformer Seed Stability",
    )

    lines = [
        "# FT-Transformer Seed Stability",
        "",
        "## Per-Seed Results",
        "",
        "| Seed | Accuracy | Macro-F1 | AUC |",
        "|---|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.accuracy:.4f} | {row.f1_macro:.4f} | {row.roc_auc_ovr_macro:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- mean accuracy = {summary['mean_accuracy']:.4f}",
            f"- std accuracy = {summary['std_accuracy']:.4f}",
            f"- mean macro-F1 = {summary['mean_f1_macro']:.4f}",
            f"- std macro-F1 = {summary['std_f1_macro']:.4f}",
            f"- mean AUC = {summary['mean_auc']:.4f}",
            f"- std AUC = {summary['std_auc']:.4f}",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(df.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved stability csv: {output_csv}")
    print(f"saved stability markdown: {output_md}")
    print(f"saved stability figure: {output_png}")


if __name__ == "__main__":
    main()
