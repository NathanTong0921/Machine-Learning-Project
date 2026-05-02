from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ml_project_matplotlib_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/ml_project_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "processed_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGET_MAP = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate",
}


def save_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    target_counts = (
        df["Target"]
        .map(TARGET_MAP)
        .value_counts()
        .rename_axis("Target")
        .reset_index(name="Count")
    )
    target_counts["Percent"] = target_counts["Count"] / len(df) * 100

    plt.figure(figsize=(7, 4.5))
    ax = sns.barplot(
        data=target_counts,
        x="Target",
        y="Count",
        hue="Target",
        palette="Set2",
        legend=False,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3)
    ax.set_title("Target Class Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Number of students")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=180)
    plt.close()

    target_counts.to_csv(OUTPUT_DIR / "target_distribution.csv", index=False)
    return target_counts


def save_feature_histograms(df: pd.DataFrame) -> None:
    feature_cols = [col for col in df.columns if col != "Target"]
    ncols = 4
    nrows = (len(feature_cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3))
    axes = axes.flatten()

    for ax, col in zip(axes, feature_cols):
        ax.hist(df[col].dropna(), bins=30, edgecolor="black", color="#4C78A8", alpha=0.75)
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[len(feature_cols) :]:
        ax.axis("off")

    fig.suptitle("Feature Distributions", fontsize=14, y=1.002)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_histograms.png", dpi=180)
    plt.close()


def save_boxplots(df: pd.DataFrame) -> None:
    feature_cols = [col for col in df.columns if col != "Target"]
    long_df = df[feature_cols].melt(var_name="Feature", value_name="Value")

    plt.figure(figsize=(18, 8))
    ax = sns.boxplot(data=long_df, x="Feature", y="Value", color="#72B7B2", fliersize=1.5)
    ax.set_title("Feature Value Ranges and Outliers")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_boxplots.png", dpi=180)
    plt.close()


def save_target_correlation(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    target_corr = (
        corr["Target"]
        .drop("Target")
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .head(15)
        .rename_axis("Feature")
        .reset_index(name="Correlation with Target")
    )

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=target_corr,
        y="Feature",
        x="Correlation with Target",
        hue="Feature",
        palette="vlag",
        legend=False,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 15 Features Correlated with Target")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_target_correlations.png", dpi=180)
    plt.close()

    target_corr.to_csv(OUTPUT_DIR / "top_target_correlations.csv", index=False)
    return target_corr


def save_correlation_heatmap(df: pd.DataFrame, target_corr: pd.DataFrame) -> None:
    top_features = target_corr["Feature"].tolist() + ["Target"]
    corr_sub = df[top_features].corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=False)
    plt.title("Correlation Heatmap: Top Target-Related Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_feature_correlation_heatmap.png", dpi=180)
    plt.close()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    target_counts = save_target_distribution(df)
    save_feature_histograms(df)
    save_boxplots(df)
    target_corr = save_target_correlation(df)
    save_correlation_heatmap(df, target_corr)

    df.describe().T.to_csv(OUTPUT_DIR / "feature_summary.csv")

    print(f"Dataset shape: {df.shape}")
    print("\nTarget distribution:")
    print(target_counts.to_string(index=False, formatters={"Percent": "{:.2f}%".format}))
    print("\nTop correlations with Target:")
    print(target_corr.head(10).to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
