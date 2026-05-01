"""Visualization helpers for EDA, t-SNE, ROC, and confusion matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns


def save_confusion_matrix(cm, labels, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_metrics_table_figure(df, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(11, 2.6 + 0.45 * len(display_df)))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax.set_title(title, fontsize=14, pad=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_grid(image_paths, titles, output_path, nrows=2, ncols=2):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 11))
    axes = axes.flatten()

    for ax, image_path, title in zip(axes, image_paths, titles):
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
