"""Search a pure-ML tabular mixture-of-experts side experiment.

Router:
    Dense Appendix-aligned sklearn preprocessing -> PCA -> Gaussian Mixture Model.

Experts:
    Local LightGBM or CatBoost experts trained on hard router clusters with
    native categorical handling.

Aggregation:
    Hard routing uses the top-1 cluster expert. Soft routing weights all local
    experts by the GMM posterior probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor, split_feature_columns


LIGHTGBM_PARAMS = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "num_leaves": 63,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "min_child_samples": 25,
    "max_depth": -1,
    "subsample": 0.8,
    "max_bin": 16,
    "colsample_bytree": 0.6,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}

CATBOOST_PARAMS = {
    "loss_function": "MultiClass",
    "eval_metric": "TotalF1:average=Macro",
    "auto_class_weights": "Balanced",
    "depth": 6,
    "iterations": 500,
    "l2_leaf_reg": 9,
    "learning_rate": 0.05,
    "random_strength": 1.0,
    "random_seed": RANDOM_STATE,
    "verbose": 0,
}


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


def dense_array(matrix):
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train_df, test_df


def router_features(train_df, test_df, pca_components):
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])
    numeric_cols, cat_cols = split_feature_columns(X_train.columns)
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    train_dense = dense_array(preprocessor.fit_transform(X_train))
    test_dense = dense_array(preprocessor.transform(X_test))

    n_components = min(pca_components, train_dense.shape[0] - 1, train_dense.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    train_router = pca.fit_transform(train_dense)
    test_router = pca.transform(test_dense)
    return train_router, test_router, {
        "numeric_dim": len(numeric_cols),
        "categorical_onehot_dim": int(train_dense.shape[1] - len(numeric_cols)),
        "dense_dim": int(train_dense.shape[1]),
        "pca_components": int(n_components),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def prepare_native_xy(df, expert_type):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].astype(str).copy()
    cat_cols = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    if expert_type == "lightgbm":
        for col in cat_cols:
            X[col] = X[col].astype("category")
        return X, y, cat_cols
    if expert_type == "catboost":
        for col in cat_cols:
            X[col] = X[col].astype(str)
        cat_idx = [X.columns.get_loc(col) for col in cat_cols]
        return X, y, cat_idx
    raise ValueError(f"Unknown expert_type: {expert_type}")


def make_expert(expert_type, n_jobs, seed):
    if expert_type == "lightgbm":
        return LGBMClassifier(**{**LIGHTGBM_PARAMS, "n_jobs": n_jobs, "random_state": seed})
    if expert_type == "catboost":
        return CatBoostClassifier(**{**CATBOOST_PARAMS, "thread_count": n_jobs, "random_seed": seed})
    raise ValueError(f"Unknown expert_type: {expert_type}")


def align_proba(proba, classes):
    aligned = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    class_to_col = {str(label): idx for idx, label in enumerate(classes)}
    for out_idx, label in enumerate(LABELS):
        if label in class_to_col:
            aligned[:, out_idx] = proba[:, class_to_col[label]]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sums, out=np.full_like(aligned, 1.0 / len(LABELS)), where=row_sums > 0)


def fit_expert(expert_type, train_subset, n_jobs, seed):
    X, y, cat_spec = prepare_native_xy(train_subset, expert_type)
    model = make_expert(expert_type, n_jobs=n_jobs, seed=seed)
    if expert_type == "lightgbm":
        model.fit(X, y, categorical_feature=cat_spec)
    else:
        model.fit(X, y, cat_features=cat_spec)
    return model


def predict_expert(model, expert_type, df):
    X, _, _ = prepare_native_xy(df, expert_type)
    return align_proba(model.predict_proba(X), model.classes_)


def evaluate_proba(name, y_true, proba):
    pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
    metrics = classification_metrics(y_true, pred, y_prob=proba, labels=LABELS)
    report = classification_report(y_true, pred, labels=LABELS, output_dict=True, zero_division=0)
    for label in LABELS:
        key = label.lower()
        metrics[f"{key}_precision"] = report[label]["precision"]
        metrics[f"{key}_recall"] = report[label]["recall"]
        metrics[f"{key}_f1"] = report[label]["f1-score"]
    metrics["model"] = name
    return json_safe(metrics)


def cluster_summary(train_df, assignments, posterior, min_cluster_size):
    rows = []
    for cluster_id in range(posterior.shape[1]):
        mask = assignments == cluster_id
        subset = train_df.loc[mask, TARGET_COL].astype(str)
        counts = subset.value_counts().reindex(LABELS, fill_value=0)
        rows.append({
            "cluster": cluster_id,
            "n": int(mask.sum()),
            "load_share": float(mask.mean()),
            "too_small": bool(mask.sum() < min_cluster_size),
            "dropout": int(counts["Dropout"]),
            "enrolled": int(counts["Enrolled"]),
            "graduate": int(counts["Graduate"]),
            "enrolled_share": float(counts["Enrolled"] / max(mask.sum(), 1)),
            "mean_router_confidence": float(posterior[mask, cluster_id].mean()) if mask.any() else 0.0,
        })
    return rows


def run_config(train_df, test_df, train_router, test_router, router_meta, k, covariance_type, expert_type, routing, n_jobs, min_cluster_size):
    y_test = test_df[TARGET_COL].astype(str).to_numpy()
    gmm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=1e-5,
        n_init=3,
        random_state=RANDOM_STATE,
    )
    train_assign = gmm.fit_predict(train_router)
    train_post = gmm.predict_proba(train_router)
    test_assign = gmm.predict(test_router)
    test_post = gmm.predict_proba(test_router)

    global_expert = fit_expert(expert_type, train_df, n_jobs=n_jobs, seed=RANDOM_STATE)
    experts = {}
    fallback_clusters = []
    for cluster_id in range(k):
        mask = train_assign == cluster_id
        if mask.sum() < min_cluster_size:
            experts[cluster_id] = global_expert
            fallback_clusters.append(cluster_id)
            continue
        experts[cluster_id] = fit_expert(
            expert_type,
            train_df.loc[mask].reset_index(drop=True),
            n_jobs=n_jobs,
            seed=RANDOM_STATE + cluster_id + 1,
        )

    if routing == "hard":
        proba = np.zeros((len(test_df), len(LABELS)), dtype=float)
        for cluster_id in range(k):
            mask = test_assign == cluster_id
            if mask.any():
                proba[mask] = predict_expert(experts[cluster_id], expert_type, test_df.loc[mask].reset_index(drop=True))
    elif routing == "soft":
        expert_probas = []
        for cluster_id in range(k):
            expert_probas.append(predict_expert(experts[cluster_id], expert_type, test_df))
        stacked = np.stack(expert_probas, axis=1)
        proba = (stacked * test_post[:, :, None]).sum(axis=1)
    else:
        raise ValueError(f"Unknown routing: {routing}")

    metrics = evaluate_proba(f"gmm_moe_{expert_type}_{routing}_k{k}_{covariance_type}", y_test, proba)
    summary_rows = cluster_summary(train_df, train_assign, train_post, min_cluster_size)
    load_shares = [row["load_share"] for row in summary_rows]
    metrics.update({
        "router": "gmm_pca",
        "expert_type": expert_type,
        "routing": routing,
        "k": k,
        "covariance_type": covariance_type,
        "fallback_clusters": json.dumps(fallback_clusters),
        "n_fallback_clusters": len(fallback_clusters),
        "min_cluster_size": min_cluster_size,
        "min_load_share": float(min(load_shares)),
        "max_load_share": float(max(load_shares)),
        "load_std": float(np.std(load_shares)),
        "router_meta_json": json.dumps(router_meta, sort_keys=True),
        "cluster_summary_json": json.dumps(summary_rows, sort_keys=True),
    })
    return metrics, summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke-test grid.")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--min-cluster-size", type=int, default=120)
    args = parser.parse_args()

    train_df, test_df = load_data()
    train_router, test_router, router_meta = router_features(train_df, test_df, args.pca_components)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    k_grid = [2, 3] if args.quick else [2, 3, 4, 5]
    covariance_grid = ["diag"] if args.quick else ["diag", "full"]
    expert_grid = ["lightgbm"] if args.quick else ["lightgbm", "catboost"]
    routing_grid = ["hard", "soft"]

    metric_records = []
    cluster_records = []
    total = len(k_grid) * len(covariance_grid) * len(expert_grid) * len(routing_grid)
    idx = 0
    for k in k_grid:
        for covariance_type in covariance_grid:
            for expert_type in expert_grid:
                for routing in routing_grid:
                    idx += 1
                    print(
                        f"[MOE] {idx}/{total} k={k} covariance={covariance_type} "
                        f"expert={expert_type} routing={routing}",
                        flush=True,
                    )
                    metrics, summary_rows = run_config(
                        train_df,
                        test_df,
                        train_router,
                        test_router,
                        router_meta,
                        k=k,
                        covariance_type=covariance_type,
                        expert_type=expert_type,
                        routing=routing,
                        n_jobs=args.n_jobs,
                        min_cluster_size=args.min_cluster_size,
                    )
                    metric_records.append(metrics)
                    for row in summary_rows:
                        cluster_records.append({
                            "model": metrics["model"],
                            "expert_type": expert_type,
                            "routing": routing,
                            "k": k,
                            "covariance_type": covariance_type,
                            **row,
                        })
                    save_metrics_table(metric_records, METRICS_DIR / "tabular_moe_metrics.csv")
                    save_metrics_table(cluster_records, METRICS_DIR / "tabular_moe_cluster_summary.csv")

    metrics_df = pd.DataFrame(metric_records).sort_values("f1_macro", ascending=False)
    clusters_df = pd.DataFrame(cluster_records)
    metrics_df.to_csv(METRICS_DIR / "tabular_moe_metrics.csv", index=False)
    clusters_df.to_csv(METRICS_DIR / "tabular_moe_cluster_summary.csv", index=False)
    best = metrics_df.iloc[0].to_dict()
    (METRICS_DIR / "tabular_moe_best.json").write_text(
        json.dumps(json_safe(best), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(metrics_df[[
        "model",
        "accuracy",
        "f1_macro",
        "roc_auc_ovr_macro",
        "dropout_accuracy",
        "enrolled_accuracy",
        "graduate_accuracy",
        "min_load_share",
        "max_load_share",
    ]].head(10).to_string(index=False))
    print(f"saved metrics: {METRICS_DIR / 'tabular_moe_metrics.csv'}")
    print(f"saved cluster summary: {METRICS_DIR / 'tabular_moe_cluster_summary.csv'}")


if __name__ == "__main__":
    main()
