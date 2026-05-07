"""Side analysis for where Enrolled samples sit between the two endpoint classes.

Declaration:
- This is a descriptive analysis script, not a training baseline.
- It does not overwrite multiclass or cumulative-link experiment scripts.
- It estimates Enrolled geometry in two ways:
  1. Endpoint distance: Enrolled to Dropout/Graduate centroids and endpoint kNN.
  2. Two-stage distance: signed linear distances for Dropout-vs-NotDropout and
     Enrolled-vs-Graduate binary separators.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import save_metrics_table
from src.preprocessing import split_feature_columns
from src.visualization import save_metrics_table_figure


NOT_DROPOUT = "NotDropout"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return value


def dense_preprocessor(numeric_features, categorical_features):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def load_full_processed_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv").copy()
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv").copy()
    train_df["split"] = "train"
    test_df["split"] = "test"
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["row_id"] = np.arange(len(df))
    return df


def signed_lr_distance(model, X):
    margin = model.decision_function(X)
    norm = np.linalg.norm(model.coef_)
    if norm == 0:
        return margin
    return margin / norm


def distance_tag(endpoint_score, graduate_neighbor_share, score_band):
    if endpoint_score >= score_band and graduate_neighbor_share >= 0.60:
        return "graduate_like"
    if endpoint_score <= -score_band and graduate_neighbor_share <= 0.40:
        return "dropout_like"
    return "middle_or_ambiguous"


def two_stage_tag(stage_a_distance, stage_b_distance, stage_band):
    if stage_a_distance < 0:
        return "dropout_side"
    if stage_b_distance >= stage_band:
        return "graduate_side_after_not_dropout"
    if stage_b_distance <= -stage_band:
        return "enrolled_core_after_not_dropout"
    return "stage_b_boundary"


def save_scatter_plot(enrolled_df, output_path):
    color_map = {
        "graduate_like": "#1b9e77",
        "dropout_like": "#d95f02",
        "middle_or_ambiguous": "#7570b3",
    }
    colors = enrolled_df["endpoint_distance_tag"].map(color_map)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(
        enrolled_df["endpoint_centroid_score"],
        enrolled_df["endpoint_knn_graduate_share"],
        c=colors,
        alpha=0.72,
        s=28,
        edgecolors="none",
    )
    axes[0].axvline(0, color="#333333", linewidth=1, linestyle="--")
    axes[0].axhline(0.5, color="#333333", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Endpoint centroid score: positive = closer to Graduate")
    axes[0].set_ylabel("Endpoint kNN Graduate share")
    axes[0].set_title("Enrolled Between Endpoint Classes")

    axes[1].scatter(
        enrolled_df["stage_a_signed_distance"],
        enrolled_df["stage_b_signed_distance"],
        c=colors,
        alpha=0.72,
        s=28,
        edgecolors="none",
    )
    axes[1].axvline(0, color="#333333", linewidth=1, linestyle="--")
    axes[1].axhline(0, color="#333333", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Stage A signed distance: positive = NotDropout")
    axes[1].set_ylabel("Stage B signed distance: positive = Graduate")
    axes[1].set_title("Two-Stage Boundary Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbors", type=int, default=25)
    parser.add_argument("--score-band", type=float, default=0.05)
    parser.add_argument("--stage-band", type=float, default=0.10)
    args = parser.parse_args()

    df = load_full_processed_data()
    feature_cols = [col for col in df.columns if col not in {TARGET_COL, "split", "row_id"}]
    X_raw = df[feature_cols]
    y = df[TARGET_COL].copy()
    numeric_features, categorical_features = split_feature_columns(feature_cols)

    preprocessor = dense_preprocessor(numeric_features, categorical_features)
    X = preprocessor.fit_transform(X_raw)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    masks = {
        "Dropout": y == "Dropout",
        "Enrolled": y == "Enrolled",
        "Graduate": y == "Graduate",
    }
    enrolled_idx = np.flatnonzero(masks["Enrolled"].to_numpy())
    endpoint_idx = np.flatnonzero((masks["Dropout"] | masks["Graduate"]).to_numpy())
    endpoint_y = y.iloc[endpoint_idx].to_numpy()

    dropout_centroid = X[masks["Dropout"].to_numpy()].mean(axis=0)
    graduate_centroid = X[masks["Graduate"].to_numpy()].mean(axis=0)
    not_dropout_centroid = X[(y != "Dropout").to_numpy()].mean(axis=0)
    enrolled_centroid = X[masks["Enrolled"].to_numpy()].mean(axis=0)

    X_enrolled = X[enrolled_idx]
    d_dropout = pairwise_distances(X_enrolled, dropout_centroid.reshape(1, -1)).ravel()
    d_graduate = pairwise_distances(X_enrolled, graduate_centroid.reshape(1, -1)).ravel()
    endpoint_score = (d_dropout - d_graduate) / np.maximum(d_dropout + d_graduate, 1e-12)

    d_not_dropout = pairwise_distances(X_enrolled, not_dropout_centroid.reshape(1, -1)).ravel()
    d_enrolled_centroid = pairwise_distances(X_enrolled, enrolled_centroid.reshape(1, -1)).ravel()
    stage_a_centroid_score = (d_dropout - d_not_dropout) / np.maximum(d_dropout + d_not_dropout, 1e-12)
    stage_b_centroid_score = (d_enrolled_centroid - d_graduate) / np.maximum(d_enrolled_centroid + d_graduate, 1e-12)

    nn = NearestNeighbors(n_neighbors=args.neighbors, metric="euclidean")
    nn.fit(X[endpoint_idx])
    _, endpoint_neighbor_pos = nn.kneighbors(X_enrolled)
    neighbor_labels = endpoint_y[endpoint_neighbor_pos]
    endpoint_knn_graduate_share = (neighbor_labels == "Graduate").mean(axis=1)
    endpoint_knn_dropout_share = (neighbor_labels == "Dropout").mean(axis=1)

    stage_a_y = np.where(y == "Dropout", "Dropout", NOT_DROPOUT)
    stage_a = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    stage_a.fit(X, stage_a_y)
    stage_a_positive = np.where(stage_a.classes_ == NOT_DROPOUT)[0][0]
    stage_a_sign = 1.0 if stage_a_positive == 1 else -1.0
    stage_a_distance = stage_a_sign * signed_lr_distance(stage_a, X_enrolled)

    not_dropout_mask = y != "Dropout"
    stage_b = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    stage_b.fit(X[not_dropout_mask.to_numpy()], y[not_dropout_mask])
    stage_b_positive = np.where(stage_b.classes_ == "Graduate")[0][0]
    stage_b_sign = 1.0 if stage_b_positive == 1 else -1.0
    stage_b_distance = stage_b_sign * signed_lr_distance(stage_b, X_enrolled)

    enrolled_rows = df.iloc[enrolled_idx].copy()
    enrolled_rows["distance_to_dropout_centroid"] = d_dropout
    enrolled_rows["distance_to_graduate_centroid"] = d_graduate
    enrolled_rows["endpoint_centroid_score"] = endpoint_score
    enrolled_rows["endpoint_knn_dropout_share"] = endpoint_knn_dropout_share
    enrolled_rows["endpoint_knn_graduate_share"] = endpoint_knn_graduate_share
    enrolled_rows["distance_to_not_dropout_centroid"] = d_not_dropout
    enrolled_rows["distance_to_enrolled_centroid"] = d_enrolled_centroid
    enrolled_rows["stage_a_centroid_score"] = stage_a_centroid_score
    enrolled_rows["stage_b_centroid_score"] = stage_b_centroid_score
    enrolled_rows["stage_a_signed_distance"] = stage_a_distance
    enrolled_rows["stage_b_signed_distance"] = stage_b_distance
    enrolled_rows["endpoint_distance_tag"] = [
        distance_tag(score, share, args.score_band)
        for score, share in zip(endpoint_score, endpoint_knn_graduate_share)
    ]
    enrolled_rows["two_stage_distance_tag"] = [
        two_stage_tag(a_dist, b_dist, args.stage_band)
        for a_dist, b_dist in zip(stage_a_distance, stage_b_distance)
    ]

    summary_records = []
    for tag_col in ["endpoint_distance_tag", "two_stage_distance_tag"]:
        counts = enrolled_rows[tag_col].value_counts().sort_index()
        for tag, count in counts.items():
            subset = enrolled_rows[enrolled_rows[tag_col] == tag]
            summary_records.append(
                {
                    "tag_type": tag_col,
                    "tag": tag,
                    "count": int(count),
                    "share": float(count / len(enrolled_rows)),
                    "mean_endpoint_centroid_score": float(subset["endpoint_centroid_score"].mean()),
                    "mean_knn_graduate_share": float(subset["endpoint_knn_graduate_share"].mean()),
                    "mean_stage_a_signed_distance": float(subset["stage_a_signed_distance"].mean()),
                    "mean_stage_b_signed_distance": float(subset["stage_b_signed_distance"].mean()),
                }
            )

    quantile_cols = [
        "distance_to_dropout_centroid",
        "distance_to_graduate_centroid",
        "endpoint_centroid_score",
        "endpoint_knn_graduate_share",
        "stage_a_signed_distance",
        "stage_b_signed_distance",
        "stage_a_centroid_score",
        "stage_b_centroid_score",
    ]
    quantile_records = []
    for col in quantile_cols:
        values = enrolled_rows[col]
        quantile_records.append(
            {
                "metric": col,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "q10": float(values.quantile(0.10)),
                "q25": float(values.quantile(0.25)),
                "q50": float(values.quantile(0.50)),
                "q75": float(values.quantile(0.75)),
                "q90": float(values.quantile(0.90)),
            }
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    enrolled_path = METRICS_DIR / "enrolled_feature_space_distances.csv"
    summary_path = METRICS_DIR / "enrolled_feature_space_distance_summary.csv"
    quantiles_path = METRICS_DIR / "enrolled_feature_space_distance_quantiles.csv"
    config_path = METRICS_DIR / "enrolled_feature_space_distance_config.json"

    enrolled_rows.to_csv(enrolled_path, index=False)
    save_metrics_table(summary_records, summary_path)
    save_metrics_table(quantile_records, quantiles_path)
    save_metrics_table_figure(
        pd.DataFrame(summary_records),
        FIGURE_DIR / "enrolled_feature_space_distance_summary.png",
        "Enrolled Feature-Space Distance Tags",
    )
    save_scatter_plot(enrolled_rows, FIGURE_DIR / "enrolled_feature_space_distance_scatter.png")
    config = {
        "n_enrolled": int(len(enrolled_rows)),
        "n_features_after_encoding": int(X.shape[1]),
        "neighbors": args.neighbors,
        "score_band": args.score_band,
        "stage_band": args.stage_band,
        "endpoint_centroid_score": "(distance_to_dropout - distance_to_graduate) / sum; positive means closer to Graduate.",
        "stage_a_signed_distance": "Positive side means NotDropout in Dropout-vs-NotDropout logistic boundary.",
        "stage_b_signed_distance": "Positive side means Graduate in Enrolled-vs-Graduate logistic boundary.",
    }
    config_path.write_text(json.dumps(json_safe(config), ensure_ascii=False, indent=2), encoding="utf-8")

    print(pd.DataFrame(summary_records).to_string(index=False))
    print("\n=== Quantiles ===")
    print(pd.DataFrame(quantile_records).to_string(index=False))
    print(f"\nsaved enrolled distances: {enrolled_path}")
    print(f"saved summary: {summary_path}")
    print(f"saved quantiles: {quantiles_path}")


if __name__ == "__main__":
    main()
