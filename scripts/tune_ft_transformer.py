"""Run a small FT-Transformer tuning sweep and save ranked results.

Declaration:
- This script performs a limited, reportable FT-Transformer tuning round.
- It keeps the search intentionally small so the comparison remains interpretable.

Tips:
- Run `scripts/prepare_data.py` first.
- Prefer macro-F1 and AUC when choosing the best FT-Transformer trial.
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


FT_TRIALS = [
    {
        "trial_name": "ft_wider_tokens",
        "seed": 62,
        "d_token": 48,
        "n_blocks": 2,
        "attention_n_heads": 4,
        "attention_dropout": 0.20,
        "ff_dropout": 0.20,
        "lr": 8e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
    },
    {
        "trial_name": "ft_token64",
        "seed": 62,
        "d_token": 64,
        "n_blocks": 2,
        "attention_n_heads": 4,
        "attention_dropout": 0.20,
        "ff_dropout": 0.20,
        "lr": 8e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
    },
    {
        "trial_name": "ft_token64_lower_dropout",
        "seed": 62,
        "d_token": 64,
        "n_blocks": 2,
        "attention_n_heads": 4,
        "attention_dropout": 0.10,
        "ff_dropout": 0.10,
        "lr": 8e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
    },
    {
        "trial_name": "ft_token64_small_lr",
        "seed": 62,
        "d_token": 64,
        "n_blocks": 2,
        "attention_n_heads": 4,
        "attention_dropout": 0.15,
        "ff_dropout": 0.15,
        "lr": 6e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
    },
    {
        "trial_name": "ft_token80_small_lr",
        "seed": 62,
        "d_token": 80,
        "n_blocks": 2,
        "attention_n_heads": 4,
        "attention_dropout": 0.15,
        "ff_dropout": 0.15,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
    },
    {
        "trial_name": "ft_token80_lower_dropout",
        "seed": 62,
        "d_token": 80,
        "n_blocks": 2,
        "attention_n_heads": 4,
        "attention_dropout": 0.10,
        "ff_dropout": 0.10,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
    },
]

FT_MAX_EPOCHS = 90
FT_PATIENCE = 16


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    records = []
    best_record = None
    best_metadata = None

    for trial in FT_TRIALS:
        result = train_ft_transformer(
            train_df=train_df,
            test_df=test_df,
            seed=trial["seed"],
            d_token=trial["d_token"],
            n_blocks=trial["n_blocks"],
            attention_n_heads=trial["attention_n_heads"],
            attention_dropout=trial["attention_dropout"],
            ff_dropout=trial["ff_dropout"],
            lr=trial["lr"],
            weight_decay=trial["weight_decay"],
            batch_size=trial["batch_size"],
            max_epochs=FT_MAX_EPOCHS,
            patience=FT_PATIENCE,
        )
        metrics = classification_metrics(
            result["test_true_labels"],
            result["test_pred_labels"],
            y_prob=result["test_prob"],
            labels=LABELS,
        )
        record = {
            **trial,
            **metrics,
        }
        records.append(record)

        if best_record is None or record["f1_macro"] > best_record["f1_macro"]:
            best_record = record
            best_metadata = result["metadata"]

    df = pd.DataFrame(records).sort_values(
        ["f1_macro", "roc_auc_ovr_macro", "accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    output_csv = METRICS_DIR / "ft_tuning_results.csv"
    output_json = METRICS_DIR / "ft_tuning_best.json"
    output_md = METRICS_DIR / "ft_tuning_ranked.md"
    output_png = FIGURE_DIR / "ft_tuning_table.png"

    save_metrics_table(df.to_dict(orient="records"), output_csv)
    save_metrics_table_figure(
        df[
            [
                "trial_name",
                "d_token",
                "n_blocks",
                "attention_dropout",
                "ff_dropout",
                "lr",
                "accuracy",
                "f1_macro",
                "roc_auc_ovr_macro",
            ]
        ],
        output_png,
        "FT-Transformer Tuning Summary",
    )

    ranked_lines = [
        "# FT-Transformer Tuning Ranked",
        "",
        "| Rank | Trial | d_token | blocks | attn_do | ff_do | lr | Macro-F1 | AUC | Accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        ranked_lines.append(
            f"| {idx} | {row.trial_name} | {row.d_token} | {row.n_blocks} | "
            f"{row.attention_dropout:.2f} | {row.ff_dropout:.2f} | {row.lr:.4f} | "
            f"{row.f1_macro:.4f} | {row.roc_auc_ovr_macro:.4f} | {row.accuracy:.4f} |"
        )
    output_md.write_text("\n".join(ranked_lines), encoding="utf-8")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_record": best_record,
                "best_metadata": best_metadata,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(df.to_string(index=False))
    print(f"saved tuning csv: {output_csv}")
    print(f"saved tuning markdown: {output_md}")
    print(f"saved tuning figure: {output_png}")
    print(f"best FT trial: {best_record['trial_name']} with macro-F1={best_record['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
