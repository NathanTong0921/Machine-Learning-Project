"""Summarize model results into a comparison table and confusion-matrix grid.

Declaration:
- This script is the reporting bridge between raw experiment outputs and report-ready assets.
- It does not retrain models; it consolidates metrics and existing confusion-matrix figures.

Tips:
- Run baseline and advanced scripts before this script.
- Use the generated CSV and PNG files directly in the report and slides.
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, METRICS_DIR
from src.visualization import save_confusion_matrix_grid, save_metrics_table_figure


def main():
    baseline_path = METRICS_DIR / "classification_metrics.csv"
    advanced_path = METRICS_DIR / "advanced_metrics.csv"
    if not baseline_path.exists() or not advanced_path.exists():
        raise FileNotFoundError(
            "Missing metrics files. Run `python scripts/baseline/run_baselines.py` and "
            "`python scripts/mlp/run_advanced.py` first."
        )

    baseline_df = pd.read_csv(baseline_path)
    advanced_df = pd.read_csv(advanced_path)
    combined_df = pd.concat([baseline_df, advanced_df], ignore_index=True)

    ordered_models = [
        "decision_tree",
        "random_forest",
        "svm_rbf",
        "multiview_gated_mlp",
    ]
    combined_df["model"] = pd.Categorical(combined_df["model"], categories=ordered_models, ordered=True)
    combined_df = combined_df.sort_values("model").reset_index(drop=True)

    output_csv = METRICS_DIR / "model_comparison_summary.csv"
    combined_df.to_csv(output_csv, index=False)

    display_columns = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
    ]
    save_metrics_table_figure(
        combined_df[display_columns],
        FIGURE_DIR / "model_comparison_table.png",
        "Model Comparison Summary",
    )

    image_paths = [
        FIGURE_DIR / "cm_decision_tree.png",
        FIGURE_DIR / "cm_random_forest.png",
        FIGURE_DIR / "cm_svm_rbf.png",
        FIGURE_DIR / "cm_multiview_gated_mlp.png",
    ]
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing confusion matrix image: {image_path}")

    save_confusion_matrix_grid(
        image_paths=image_paths,
        titles=[
            "Decision Tree",
            "Random Forest",
            "SVM (RBF)",
            "Multi-View Gated MLP",
        ],
        output_path=FIGURE_DIR / "confusion_matrix_grid.png",
    )

    print(f"saved summary csv: {output_csv}")
    print(f"saved table figure: {FIGURE_DIR / 'model_comparison_table.png'}")
    print(f"saved confusion grid: {FIGURE_DIR / 'confusion_matrix_grid.png'}")


if __name__ == "__main__":
    main()
