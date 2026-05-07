"""Evaluate ordinal-style cumulative binary decompositions.

Declaration:
- This script appends a new experiment and does not overwrite the original
  multiclass training scripts.
- It treats the target as ordered: Dropout < Enrolled < Graduate.
- Each member trains two binary classifiers:
  A: Dropout vs NotDropout.
  B: Enrolled vs Graduate, fitted only on non-dropout training rows.
- The two binary probabilities are recomposed into three-class probabilities.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

if not hasattr(np, "NaN"):
    np.NaN = np.nan

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns
from src.visualization import save_metrics_table_figure


WEIGHT_STEP = 0.05
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def prepare_catboost_xy(df):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    cat_cols = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    cat_idx = [X.columns.get_loc(col) for col in cat_cols]
    for col in cat_cols:
        X[col] = X[col].astype(str)
    return X, y, cat_idx


def prepare_lightgbm_xy(df):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    cat_cols = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    for col in cat_cols:
        X[col] = X[col].astype("category")
    return X, y, cat_cols


def binary_proba(model, X, positive_label):
    proba = model.predict_proba(X)
    class_to_idx = {str(label): idx for idx, label in enumerate(model.classes_)}
    return proba[:, class_to_idx[positive_label]]


def combine_binary_probabilities(p_not_dropout, p_graduate_given_not_dropout):
    p_not_dropout = np.clip(p_not_dropout, 0.0, 1.0)
    p_graduate_given_not_dropout = np.clip(p_graduate_given_not_dropout, 0.0, 1.0)
    proba = np.column_stack(
        [
            1.0 - p_not_dropout,
            p_not_dropout * (1.0 - p_graduate_given_not_dropout),
            p_not_dropout * p_graduate_given_not_dropout,
        ]
    )
    row_sum = proba.sum(axis=1, keepdims=True)
    return proba / np.where(row_sum == 0, 1.0, row_sum)


def evaluate(name, y_true, proba):
    pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
    metrics = classification_metrics(y_true, pred, y_prob=proba, labels=LABELS)
    report = classification_report(y_true, pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["model"] = name
    for label in LABELS:
        key = label.lower()
        metrics[f"{key}_precision"] = report[label]["precision"]
        metrics[f"{key}_recall"] = report[label]["recall"]
        metrics[f"{key}_f1"] = report[label]["f1-score"]
    return metrics


def evaluate_binary_stages(name, y_true, p_not_dropout, p_graduate_given_not_dropout):
    records = []
    y_stage_a = np.where(y_true == "Dropout", "Dropout", NOT_DROPOUT)
    pred_stage_a = np.where(p_not_dropout >= 0.5, NOT_DROPOUT, "Dropout")
    records.append(
        {
            "model": f"{name}_stage_a_dropout_vs_not",
            "positive_label": NOT_DROPOUT,
            "f1_macro": f1_score(y_stage_a, pred_stage_a, average="macro", zero_division=0),
            "auc": roc_auc_score((y_stage_a == NOT_DROPOUT).astype(int), p_not_dropout),
        }
    )

    mask = y_true != "Dropout"
    y_stage_b = np.asarray(y_true[mask])
    pred_stage_b = np.where(p_graduate_given_not_dropout[mask] >= 0.5, "Graduate", "Enrolled")
    records.append(
        {
            "model": f"{name}_stage_b_enrolled_vs_graduate",
            "positive_label": "Graduate",
            "f1_macro": f1_score(y_stage_b, pred_stage_b, average="macro", zero_division=0),
            "auc": roc_auc_score((y_stage_b == "Graduate").astype(int), p_graduate_given_not_dropout[mask]),
        }
    )
    return records


def make_catboost_binary(params, seed):
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="F1",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=seed,
        thread_count=-1,
        **params,
    )


def make_lightgbm_binary(params, seed, n_jobs):
    return LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
        n_jobs=n_jobs,
        **params,
    )


def make_gbdt_binary(params, seed, numeric_features, categorical_features):
    model = GradientBoostingClassifier(loss="log_loss", random_state=seed)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", dense_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )
    pipeline.set_params(**params)
    return pipeline


def fit_predict_catboost(params, train_df, test_df, cv):
    X, y, cat_idx = prepare_catboost_xy(train_df)
    X_test, _, _ = prepare_catboost_xy(test_df)
    p_not_dropout_oof = np.zeros(len(train_df), dtype=float)
    p_grad_oof = np.zeros(len(train_df), dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member=catboost_cumulative fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        y_a = np.where(y.iloc[train_idx] == "Dropout", "Dropout", NOT_DROPOUT)
        model_a = make_catboost_binary(params, RANDOM_STATE + fold_idx)
        model_a.fit(X.iloc[train_idx], y_a, cat_features=cat_idx)
        p_not_dropout_oof[valid_idx] = binary_proba(model_a, X.iloc[valid_idx], NOT_DROPOUT)

        b_train_idx = [idx for idx in train_idx if y.iloc[idx] != "Dropout"]
        model_b = make_catboost_binary(params, RANDOM_STATE + 1000 + fold_idx)
        model_b.fit(X.iloc[b_train_idx], y.iloc[b_train_idx], cat_features=cat_idx)
        p_grad_oof[valid_idx] = binary_proba(model_b, X.iloc[valid_idx], "Graduate")

    y_a_full = np.where(y == "Dropout", "Dropout", NOT_DROPOUT)
    model_a = make_catboost_binary(params, RANDOM_STATE)
    model_a.fit(X, y_a_full, cat_features=cat_idx)
    p_not_dropout_test = binary_proba(model_a, X_test, NOT_DROPOUT)

    mask = y != "Dropout"
    model_b = make_catboost_binary(params, RANDOM_STATE + 1000)
    model_b.fit(X.loc[mask], y.loc[mask], cat_features=cat_idx)
    p_grad_test = binary_proba(model_b, X_test, "Graduate")
    return combine_binary_probabilities(p_not_dropout_oof, p_grad_oof), combine_binary_probabilities(p_not_dropout_test, p_grad_test), p_not_dropout_test, p_grad_test


def fit_predict_lightgbm(name, params, train_df, test_df, cv, n_jobs):
    X, y, cat_cols = prepare_lightgbm_xy(train_df)
    X_test, _, _ = prepare_lightgbm_xy(test_df)
    p_not_dropout_oof = np.zeros(len(train_df), dtype=float)
    p_grad_oof = np.zeros(len(train_df), dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member={name} fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        y_a = pd.Series(np.where(y.iloc[train_idx] == "Dropout", "Dropout", NOT_DROPOUT), index=y.index[train_idx])
        model_a = make_lightgbm_binary(params, RANDOM_STATE + fold_idx, n_jobs)
        model_a.fit(X.iloc[train_idx], y_a, categorical_feature=cat_cols)
        p_not_dropout_oof[valid_idx] = binary_proba(model_a, X.iloc[valid_idx], NOT_DROPOUT)

        b_train_idx = [idx for idx in train_idx if y.iloc[idx] != "Dropout"]
        model_b = make_lightgbm_binary(params, RANDOM_STATE + 1000 + fold_idx, n_jobs)
        model_b.fit(X.iloc[b_train_idx], y.iloc[b_train_idx], categorical_feature=cat_cols)
        p_grad_oof[valid_idx] = binary_proba(model_b, X.iloc[valid_idx], "Graduate")

    y_a_full = pd.Series(np.where(y == "Dropout", "Dropout", NOT_DROPOUT), index=y.index)
    model_a = make_lightgbm_binary(params, RANDOM_STATE, n_jobs)
    model_a.fit(X, y_a_full, categorical_feature=cat_cols)
    p_not_dropout_test = binary_proba(model_a, X_test, NOT_DROPOUT)

    mask = y != "Dropout"
    model_b = make_lightgbm_binary(params, RANDOM_STATE + 1000, n_jobs)
    model_b.fit(X.loc[mask], y.loc[mask], categorical_feature=cat_cols)
    p_grad_test = binary_proba(model_b, X_test, "Graduate")
    return combine_binary_probabilities(p_not_dropout_oof, p_grad_oof), combine_binary_probabilities(p_not_dropout_test, p_grad_test), p_not_dropout_test, p_grad_test


def fit_predict_gbdt(params, train_df, test_df, cv, numeric_features, categorical_features):
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    p_not_dropout_oof = np.zeros(len(train_df), dtype=float)
    p_grad_oof = np.zeros(len(train_df), dtype=float)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member=gbdt_cumulative fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        y_a = pd.Series(np.where(y.iloc[train_idx] == "Dropout", "Dropout", NOT_DROPOUT), index=y.index[train_idx])
        model_a = make_gbdt_binary(params, RANDOM_STATE + fold_idx, numeric_features, categorical_features)
        model_a.fit(X.iloc[train_idx], y_a, model__sample_weight=compute_sample_weight("balanced", y_a))
        p_not_dropout_oof[valid_idx] = binary_proba(model_a, X.iloc[valid_idx], NOT_DROPOUT)

        b_train_idx = [idx for idx in train_idx if y.iloc[idx] != "Dropout"]
        y_b = y.iloc[b_train_idx]
        model_b = make_gbdt_binary(params, RANDOM_STATE + 1000 + fold_idx, numeric_features, categorical_features)
        model_b.fit(X.iloc[b_train_idx], y_b, model__sample_weight=compute_sample_weight("balanced", y_b))
        p_grad_oof[valid_idx] = binary_proba(model_b, X.iloc[valid_idx], "Graduate")

    y_a_full = pd.Series(np.where(y == "Dropout", "Dropout", NOT_DROPOUT), index=y.index)
    model_a = make_gbdt_binary(params, RANDOM_STATE, numeric_features, categorical_features)
    model_a.fit(X, y_a_full, model__sample_weight=compute_sample_weight("balanced", y_a_full))
    p_not_dropout_test = binary_proba(model_a, X_test, NOT_DROPOUT)

    mask = y != "Dropout"
    y_b = y.loc[mask]
    model_b = make_gbdt_binary(params, RANDOM_STATE + 1000, numeric_features, categorical_features)
    model_b.fit(X.loc[mask], y_b, model__sample_weight=compute_sample_weight("balanced", y_b))
    p_grad_test = binary_proba(model_b, X_test, "Graduate")
    return combine_binary_probabilities(p_not_dropout_oof, p_grad_oof), combine_binary_probabilities(p_not_dropout_test, p_grad_test), p_not_dropout_test, p_grad_test


def generate_weight_grid(n_members, step):
    units = int(round(1.0 / step))

    def rec(prefix, remaining, members_left):
        if members_left == 1:
            yield prefix + [remaining]
            return
        for value in range(remaining + 1):
            yield from rec(prefix + [value], remaining - value, members_left - 1)

    for weights in rec([], units, n_members):
        yield np.array(weights, dtype=float) / units


def search_weights(member_names, oof_probas, y_true, step):
    records = []
    best_score = -np.inf
    best_weights = None
    stacked = np.stack(oof_probas, axis=0)
    for weights in generate_weight_grid(len(member_names), step):
        proba = np.tensordot(weights, stacked, axes=(0, 0))
        pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        record = {f"weight_{name}": float(weight) for name, weight in zip(member_names, weights)}
        record["oof_f1_macro"] = float(score)
        records.append(record)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, best_score, pd.DataFrame(records).sort_values("oof_f1_macro", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--quick", action="store_true", help="Use only LightGBM hist for a smoke test.")
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    numeric_features, categorical_features = split_feature_columns(train_df.drop(columns=[TARGET_COL]).columns)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    catboost_params = load_json(METRICS_DIR / "catboost_tuned_best_params.json")
    lightgbm_params = load_json(METRICS_DIR / "lightgbm_tuned_best_params.json")
    hist = load_json(METRICS_DIR / "hist_column_sampling_best_params.json")["models"]["lightgbm"]["params"]
    classic = load_json(METRICS_DIR / "classic_boosting_best_params.json")["models"]["gbdt_log_loss"]["best_params"]
    lightgbm_hist_params = {**lightgbm_params, "max_bin": hist["max_bin"], "colsample_bytree": hist["column_fraction"]}

    members = [
        ("catboost_cumulative", "catboost", catboost_params),
        ("lightgbm_hist_cumulative", "lightgbm", lightgbm_hist_params),
        ("gbdt_cumulative", "gbdt", classic),
    ]
    if args.quick:
        members = [member for member in members if member[0] == "lightgbm_hist_cumulative"]

    metric_records = []
    stage_records = []
    member_names = []
    oof_probas = []
    test_probas = []

    for name, kind, params in members:
        if kind == "catboost":
            oof, test, p_not_dropout_test, p_grad_test = fit_predict_catboost(params, train_df, test_df, cv)
        elif kind == "lightgbm":
            oof, test, p_not_dropout_test, p_grad_test = fit_predict_lightgbm(name, params, train_df, test_df, cv, args.n_jobs)
        else:
            oof, test, p_not_dropout_test, p_grad_test = fit_predict_gbdt(
                params,
                train_df,
                test_df,
                cv,
                numeric_features,
                categorical_features,
            )
        member_names.append(name)
        oof_probas.append(oof)
        test_probas.append(test)
        metric_records.append(evaluate(f"{name}_oof", y_train, oof))
        metric_records.append(evaluate(f"{name}_test", y_test, test))
        stage_records.extend(evaluate_binary_stages(f"{name}_test", y_test.to_numpy(), p_not_dropout_test, p_grad_test))

    if len(member_names) > 1:
        weights, best_score, weight_grid = search_weights(member_names, oof_probas, y_train, WEIGHT_STEP)
        ensemble_oof = np.tensordot(weights, np.stack(oof_probas, axis=0), axes=(0, 0))
        ensemble_test = np.tensordot(weights, np.stack(test_probas, axis=0), axes=(0, 0))
        metric_records.append(evaluate("cumulative_link_ensemble_oof", y_train, ensemble_oof))
        metric_records.append(evaluate("cumulative_link_ensemble_test", y_test, ensemble_test))
        save_metrics_table(weight_grid.to_dict(orient="records"), METRICS_DIR / "cumulative_link_weight_grid.csv")
        best_payload = {
            "members": member_names,
            "weights": {name: float(weight) for name, weight in zip(member_names, weights)},
            "oof_f1_macro": float(best_score),
            "cv_splits": args.cv,
            "weight_step": WEIGHT_STEP,
            "decomposition": {
                "stage_a": "Dropout vs NotDropout",
                "stage_b": "Graduate vs Enrolled, trained only on non-dropout rows",
                "recomposition": [
                    "P(Dropout)=1-P(NotDropout)",
                    "P(Enrolled)=P(NotDropout)*(1-P(Graduate|NotDropout))",
                    "P(Graduate)=P(NotDropout)*P(Graduate|NotDropout)",
                ],
            },
        }
    else:
        best_payload = {
            "members": member_names,
            "weights": {member_names[0]: 1.0},
            "oof_f1_macro": float(metric_records[0]["f1_macro"]),
            "cv_splits": args.cv,
            "weight_step": None,
        }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_table(metric_records, METRICS_DIR / "cumulative_link_metrics.csv")
    save_metrics_table(stage_records, METRICS_DIR / "cumulative_link_binary_stage_metrics.csv")
    (METRICS_DIR / "cumulative_link_best_weights.json").write_text(
        json.dumps(json_safe(best_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_df = pd.DataFrame(metric_records)
    summary_models = [name for name in metrics_df["model"] if name.endswith("_test")]
    save_metrics_table_figure(
        metrics_df[metrics_df["model"].isin(summary_models)][
            [
                "model",
                "accuracy",
                "fmi",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
                "dropout_accuracy",
                "enrolled_accuracy",
                "graduate_accuracy",
            ]
        ],
        FIGURE_DIR / "cumulative_link_summary.png",
        "Cumulative Link Decomposition",
    )

    print(metrics_df.sort_values("f1_macro", ascending=False).to_string(index=False))
    print(f"weights: {best_payload['weights']}")


if __name__ == "__main__":
    main()
