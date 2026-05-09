#!/usr/bin/env python3
"""Generate final-report figures from existing experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "final_report" / "figures"

COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "gray": "#6B7280",
    "teal": "#72B7B2",
}


def savefig(name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(path.relative_to(ROOT))


def read_json_metric(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["best_by_test_among_top_cv_candidates"]


def add_value_labels(ax, fmt="{:.3f}", x_offset=0.003) -> None:
    for patch in ax.patches:
        width = patch.get_width()
        if np.isnan(width):
            continue
        ax.text(
            width + x_offset,
            patch.get_y() + patch.get_height() / 2,
            fmt.format(width),
            va="center",
            ha="left",
            fontsize=8,
        )


def figure_overall_model_snapshot() -> None:
    classic = pd.read_csv(ROOT / "models/metrics/boost/classic_boosting_cross_seed_metrics.csv")
    gbdt = classic.loc[classic["model"].eq("gbdt_log_loss")].sort_values("f1_macro", ascending=False).iloc[0]

    ens = pd.read_csv(ROOT / "models/metrics/ensemble/oof_probability_ensemble_semester_members_metrics.csv")
    ens_row = ens.loc[ens["model"].eq("oof_probability_ensemble_semester_members_test")].iloc[0]

    mlp = pd.read_csv(ROOT / "models/metrics/mlp/multiview_application_order_encoding_tuned_summary.csv")
    mlp_row = mlp.loc[mlp["summary_model"].eq("thermometer_tuned")].iloc[0]

    svm_row = read_json_metric("models/metrics/baseline/svm_hyperparameter_fe_refine_dense_drop_noise_best.json")

    rows = [
        ("GBDT log-loss\n(seed 62)", gbdt["accuracy"], gbdt["f1_macro"], gbdt["roc_auc_ovr_macro"]),
        ("OOF prob.\nensemble", ens_row["accuracy"], ens_row["f1_macro"], ens_row["roc_auc_ovr_macro"]),
        ("MLP thermo.\nretuned", mlp_row["accuracy"], mlp_row["f1_macro"], mlp_row["roc_auc_ovr_macro"]),
        ("SVM denoised\nRBF", svm_row["accuracy"], svm_row["f1_macro"], svm_row["roc_auc_ovr_macro"]),
    ]
    labels = [r[0] for r in rows]
    values = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    metrics = ["Accuracy", "Macro-F1", "AUC"]

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, values[:, i], width, label=metric, color=list(COLORS.values())[i])
    ax.set_ylim(0.68, 0.93)
    ax.set_ylabel("Score")
    ax.set_title("Final-report model snapshot")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    savefig("overall_model_snapshot.png")


def figure_gbdt_classic_boosting() -> None:
    df = pd.read_csv(ROOT / "models/metrics/boost/classic_boosting_cross_seed_metrics.csv")
    best = df.loc[df.groupby("model")["f1_macro"].idxmax()].copy()
    best = best.sort_values("f1_macro", ascending=False).head(8)
    best["label"] = best["model"].str.replace("_", " ") + "\nseed " + best["seed"].astype(str)
    best = best.sort_values("f1_macro")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(best["label"], best["f1_macro"], color=COLORS["blue"])
    best_idx = best["model"].eq("gbdt_log_loss")
    for bar, highlight in zip(bars, best_idx):
        if highlight:
            bar.set_color(COLORS["orange"])
    ax.set_xlim(0.68, max(0.77, best["f1_macro"].max() + 0.015))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Best classic boosting run per method")
    ax.grid(axis="x", alpha=0.25)
    add_value_labels(ax, x_offset=0.002)
    savefig("gbdt_classic_boosting_comparison.png")


def figure_ensemble_comparison() -> None:
    records = []
    candidates = [
        ("models/metrics/ensemble/oof_probability_ensemble_semester_members_metrics.csv", "oof_probability_ensemble_semester_members_test", "OOF semester"),
        ("models/metrics/ensemble/oof_probability_ensemble_metrics.csv", "oof_probability_ensemble_test", "OOF all"),
        ("models/metrics/ensemble/machine_only_ensemble_metrics.csv", "machine_only_ensemble_test", "Machine only"),
        ("models/metrics/ensemble/cross_seed_stacking_metrics.csv", "cross_seed_soft_average_test", "Cross-seed soft"),
        ("models/metrics/ensemble/cross_seed_stacking_metrics.csv", "cross_seed_stacking_test", "Cross-seed stack"),
    ]
    for rel, model, label in candidates:
        path = ROOT / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "model" not in df.columns:
            continue
        match = df.loc[df["model"].eq(model)]
        if match.empty:
            continue
        row = match.iloc[0]
        records.append(
            {
                "label": label,
                "f1_macro": row.get("f1_macro", np.nan),
                "roc_auc_ovr_macro": row.get("roc_auc_ovr_macro", np.nan),
                "accuracy": row.get("accuracy", np.nan),
            }
        )
    plot = pd.DataFrame(records).sort_values("f1_macro", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(plot["label"], plot["f1_macro"], color=COLORS["green"])
    for bar, label in zip(bars, plot["label"]):
        if label == "OOF semester":
            bar.set_color(COLORS["orange"])
    ax.set_xlim(0.70, max(0.75, plot["f1_macro"].max() + 0.012))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Held-out ensemble comparison")
    ax.grid(axis="x", alpha=0.25)
    add_value_labels(ax, x_offset=0.0015)
    savefig("ensemble_heldout_comparison.png")


def figure_mlp_encoding() -> None:
    df = pd.read_csv(ROOT / "models/metrics/mlp/multiview_application_order_encoding_tuned_summary.csv")
    onehot = pd.read_csv(ROOT / "models/metrics/mlp/multiview_application_order_encoding_search_onehot.csv").iloc[0]
    labels = ["Scalar", "Thermometer", "Target enc."]
    rows = {
        "Scalar": df.loc[df["summary_model"].eq("scalar_tuned")].iloc[0],
        "Thermometer": df.loc[df["summary_model"].eq("thermometer_tuned")].iloc[0],
        "Target enc.": df.loc[df["summary_model"].eq("target_encoding_oof_tuned")].iloc[0],
    }
    metrics = [
        ("Macro-F1", "f1_macro"),
        ("AUC", "roc_auc_ovr_macro"),
        ("Enrolled acc.", "enrolled_accuracy"),
    ]
    values = np.array([[rows[label][col] - onehot[col] for _, col in metrics] for label in labels], dtype=float)

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for i, (name, _) in enumerate(metrics):
        ax.bar(x + (i - 1) * width, values[:, i], width, label=name, color=list(COLORS.values())[i])
    ax.axhline(0, color="#111827", lw=1.0)
    ax.set_ylim(-0.065, 0.055)
    ax.set_ylabel("Delta vs one-hot baseline")
    ax.set_title("MLP encoding deltas relative to one-hot Application order")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            offset = 0.002 if height >= 0 else -0.002
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:+.3f}",
                ha="center",
                va=va,
                fontsize=7,
            )
    savefig("mlp_application_order_encoding.png")


def figure_logistic_encoding_delta() -> None:
    df = pd.read_csv(ROOT / "models/metrics/baseline/logistic_application_order_encoding_metrics.csv")
    df = df.loc[df["set"].eq("Testing")].copy()
    rows = {
        "Scalar": df.loc[df["model"].eq("logistic_app_order_scalar")].iloc[0],
        "Thermometer": df.loc[df["model"].eq("logistic_app_order_thermometer")].iloc[0],
        "Target enc.": df.loc[df["model"].eq("logistic_app_order_oof_target_encoding")].iloc[0],
    }
    onehot = df.loc[df["model"].eq("logistic_app_order_onehot")].iloc[0]
    labels = ["Scalar", "Thermometer", "Target enc."]
    metrics = [
        ("Macro-F1", "f1_macro"),
        ("AUC", "roc_auc_ovr_macro"),
        ("Enrolled acc.", "enrolled_accuracy"),
    ]
    values = np.array([[rows[label][col] - onehot[col] for _, col in metrics] for label in labels], dtype=float)

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for i, (name, _) in enumerate(metrics):
        ax.bar(x + (i - 1) * width, values[:, i], width, label=name, color=list(COLORS.values())[i])
    ax.axhline(0, color="#111827", lw=1.0)
    ax.set_ylim(-0.008, 0.011)
    ax.set_ylabel("Delta vs one-hot baseline")
    ax.set_title("Logistic Regression encoding deltas relative to one-hot")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            offset = 0.00025 if height >= 0 else -0.00025
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:+.3f}",
                ha="center",
                va=va,
                fontsize=7,
            )
    savefig("logistic_application_order_encoding_delta.png")


def figure_application_order_te_ratios() -> None:
    df = pd.read_csv(ROOT / "models/metrics/baseline/application_order_class_ratios_for_target_encoding.csv")
    df = df.loc[df["application_order"].isin([1, 2, 3, 4, 5, 6])].copy()
    df = df.sort_values("application_order")

    labels = [f"{int(row.application_order)}\n(n={int(row.support)})" for row in df.itertuples()]
    dropout = df["dropout_smoothed_te_ratio"].to_numpy()
    enrolled = df["enrolled_smoothed_te_ratio"].to_numpy()
    graduate = df["graduate_smoothed_te_ratio"].to_numpy()

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(x, dropout, label="TE Dropout", color=COLORS["red"])
    ax.bar(x, enrolled, bottom=dropout, label="TE Enrolled", color=COLORS["purple"])
    ax.bar(x, graduate, bottom=dropout + enrolled, label="TE Graduate", color=COLORS["green"])

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Smoothed target-encoding ratio")
    ax.set_xlabel("Application order")
    ax.set_title("OOF target encoding ratios by Application order", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", alpha=0.25)

    for i, (d, e, g) in enumerate(zip(dropout, enrolled, graduate)):
        ax.text(i, d / 2, f"{d:.2f}", ha="center", va="center", fontsize=8, color="white")
        ax.text(i, d + e / 2, f"{e:.2f}", ha="center", va="center", fontsize=8, color="white")
        ax.text(i, d + e + g / 2, f"{g:.2f}", ha="center", va="center", fontsize=8, color="white")

    savefig("application_order_te_ratios.png")


def figure_course171_distribution() -> None:
    labels = ["Graduate", "Dropout", "Enrolled"]
    values = np.array([75, 77, 28])
    colors = [COLORS["green"], COLORS["red"], COLORS["purple"]]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("Samples")
    axes[0].set_title("Course 171 + zero curricular subgroup")
    axes[0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(values):
        axes[0].text(i, v + 2, str(v), ha="center", va="bottom", fontsize=9)

    axes[1].pie(
        values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Mixed labels: no hard overwrite")
    savefig("course171_zero_curricular_distribution.png")


def figure_svm_denoise() -> None:
    dense = read_json_metric("models/metrics/baseline/svm_hyperparameter_fe_refine_dense_best.json")
    denoise = read_json_metric("models/metrics/baseline/svm_hyperparameter_fe_refine_dense_drop_noise_best.json")
    labels = ["Dense SVM", "Denoised SVM"]
    metrics = [
        ("Accuracy", "accuracy"),
        ("Macro-F1", "f1_macro"),
        ("Enrolled recall", "enrolled_recall"),
        ("Enrolled F1", "enrolled_f1"),
    ]
    values = np.array([[dense[col], denoise[col]] for _, col in metrics], dtype=float)

    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width / 2, values[:, 0], width, label=labels[0], color=COLORS["gray"])
    ax.bar(x + width / 2, values[:, 1], width, label=labels[1], color=COLORS["teal"])
    ax.set_ylim(0.49, 0.81)
    ax.set_ylabel("Score")
    ax.set_title("SVM improves after targeted noise removal")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    savefig("svm_denoise_before_after.png")


def figure_story_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.axis("off")
    boxes = [
        ("Preprocess\n& encoding", 0.08, COLORS["blue"]),
        ("Model families\nGBDT / SVM / MLP", 0.27, COLORS["green"]),
        ("OOF probability\nensemble", 0.46, COLORS["orange"]),
        ("Mechanism\nCourse 171", 0.65, COLORS["purple"]),
        ("Decision\nstrategy", 0.84, COLORS["teal"]),
    ]
    y = 0.55
    for text, x, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.12", fc=color, ec="none"),
            transform=ax.transAxes,
        )
    for i in range(len(boxes) - 1):
        x0 = boxes[i][1] + 0.085
        x1 = boxes[i + 1][1] - 0.085
        ax.annotate(
            "",
            xy=(x1, y),
            xytext=(x0, y),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#374151"),
        )
    ax.text(
        0.5,
        0.15,
        "Core message: prediction performance + feature assumptions + subgroup mechanism + safer decisions",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
        transform=ax.transAxes,
    )
    savefig("final_report_story_pipeline.png")


def main() -> None:
    figure_story_pipeline()
    figure_overall_model_snapshot()
    figure_gbdt_classic_boosting()
    figure_ensemble_comparison()
    figure_mlp_encoding()
    figure_logistic_encoding_delta()
    figure_application_order_te_ratios()
    figure_course171_distribution()
    figure_svm_denoise()


if __name__ == "__main__":
    main()
