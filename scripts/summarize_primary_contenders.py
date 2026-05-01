"""Summarize the main contenders into a single comparison table.

Declaration:
- This script consolidates the current strongest baselines and fancy models into one table.
- It focuses on the main contenders requested by the user.

Tips:
- This is a report-facing summary layer.
- The target set is: RF, soft voting, Multi-View Gated MLP, TabPFN, and CatBoost.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, METRICS_DIR
from src.visualization import save_metrics_table_figure


def load_row(path, model_name=None):
    df = pd.read_csv(path)
    if model_name is None:
        return df.iloc[0].to_dict()
    row = df[df["model"] == model_name].iloc[0]
    return row.to_dict()


def main():
    rows = [
        load_row(METRICS_DIR / "medium_baseline_gridsearch_metrics.csv", "random_forest"),
        load_row(METRICS_DIR / "medium_baseline_ensemble_metrics.csv", "voting_soft"),
        load_row(METRICS_DIR / "advanced_model_comparison.csv", "multiview_gated_mlp_best"),
        load_row(METRICS_DIR / "tabpfn_metrics.csv"),
        load_row(METRICS_DIR / "catboost_metrics.csv"),
        load_row(METRICS_DIR / "catboost_tuned_metrics.csv"),
    ]

    df = pd.DataFrame(rows)
    output_csv = METRICS_DIR / "primary_contenders_summary.csv"
    df.to_csv(output_csv, index=False)

    save_metrics_table_figure(
        df[
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
        FIGURE_DIR / "primary_contenders_summary.png",
        "Primary Contenders Summary",
    )

    print(df.to_string(index=False))
    print(f"saved summary csv: {output_csv}")
    print(f"saved summary figure: {FIGURE_DIR / 'primary_contenders_summary.png'}")


if __name__ == "__main__":
    main()
