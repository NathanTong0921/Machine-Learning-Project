#!/usr/bin/env python3
"""Evaluate core reproducible models across stratified seeds and summarize.

This script uses the processed train/test CSVs as one full processed dataset,
then repeatedly creates stratified 70/30 splits for each seed. It is intended
for report-level stability checks, not as a replacement for the original fixed
train/test leaderboard.
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
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

if not hasattr(np, "NaN"):
    np.NaN = np.nan

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
ENSEMBLE_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "ensemble"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENSEMBLE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ENSEMBLE_SCRIPT_DIR))

from src.baselines import get_baseline_models
from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_metrics
from src.preprocessing import build_preprocessor, split_feature_columns
from scripts.baseline.search_svm_hyperparameters import (
    NOISE_DROP_COLUMNS,
    make_preprocessor as make_svm_preprocessor,
)
import search_supervised_error_router_moe as moe_base
import search_supervised_error_router_moe_v2 as moe_v2
import tune_machine_only_ensemble as machine_only


BOOST_DIR = METRICS_DIR / "boost"
BAGGING_DIR = METRICS_DIR / "bagging"
BASELINE_DIR = METRICS_DIR / "baseline"
REPORTING_DIR = METRICS_DIR / "reporting"
SEED_LEVEL_OUT = REPORTING_DIR / "all_model_results_cross_seed_seed_level.csv"
SUMMARY_OUT = REPORTING_DIR / "all_model_results_cross_seed_summary.csv"

DEFAULT_SEEDS = [42, 62, 82, 102, 122]

METRIC_MAP = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision macro",
    "precision_weighted": "Precision weighted",
    "recall_macro": "Recall macro",
    "recall_weighted": "Recall weighted",
    "f1_macro": "F1 macro",
    "f1_weighted": "F1 weighted",
    "roc_auc_ovr_macro": "AUC",
    "fmi": "FMI",
    "dropout_precision": "Dropout precision",
    "dropout_recall": "Dropout recall",
    "dropout_f1": "Dropout F1",
    "dropout_accuracy": "Dropout acc",
    "enrolled_precision": "Enrolled precision",
    "enrolled_recall": "Enrolled recall",
    "enrolled_f1": "Enrolled F1",
    "enrolled_accuracy": "Enrolled acc",
    "graduate_precision": "Graduate precision",
    "graduate_recall": "Graduate recall",
    "graduate_f1": "Graduate F1",
    "graduate_accuracy": "Graduate acc",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def strip_model_prefix(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in params.items():
        out[key.removeprefix("model__")] = value
    return out


def split_nested_estimator_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_params = {}
    estimator_params = {}
    for key, value in strip_model_prefix(params).items():
        if key.startswith("estimator__"):
            estimator_params[key.removeprefix("estimator__")] = value
        else:
            model_params[key] = value
    return model_params, estimator_params


def load_full_processed_data() -> tuple[pd.DataFrame, pd.Series]:
    frames = [pd.read_csv(PROCESSED_DIR / "train.csv"), pd.read_csv(PROCESSED_DIR / "test.csv")]
    df = pd.concat(frames, ignore_index=True)
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def align_prob(prob: np.ndarray | None, classes) -> np.ndarray | None:
    if prob is None:
        return None
    classes = list(classes)
    if classes == LABELS:
        return prob
    aligned = np.zeros((prob.shape[0], len(LABELS)), dtype=float)
    for idx, label in enumerate(classes):
        if label in LABELS:
            aligned[:, LABELS.index(label)] = prob[:, idx]
    return aligned


def evaluate_record(model_name: str, family: str, seed: int, y_true, y_pred, y_prob, source: str) -> dict[str, Any]:
    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, labels=LABELS)
    return {
        "model": model_name,
        "family": family,
        "set": "Testing",
        "seed": seed,
        "source": source,
        **metrics,
    }


def pipeline_predict_record(model_name, family, seed, pipeline, X_train, y_train, X_test, y_test, source, sample_weight=False):
    fit_kwargs = {}
    if sample_weight:
        from sklearn.utils.class_weight import compute_sample_weight

        fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
    pipeline.fit(X_train, y_train, **fit_kwargs)
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = align_prob(pipeline.predict_proba(X_test), pipeline.classes_)
    return evaluate_record(model_name, family, seed, y_test, y_pred, y_prob, source)


def prepare_native_category(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_native = X_train.copy()
    X_test_native = X_test.copy()
    cat_cols = [col for col in X_train.columns if col in CATEGORICAL_COLUMNS]
    for col in cat_cols:
        X_train_native[col] = X_train_native[col].astype("category")
        X_test_native[col] = X_test_native[col].astype("category")
    return X_train_native, X_test_native, cat_cols


def prepare_catboost(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    cat_cols = [col for col in X_train.columns if col in CATEGORICAL_COLUMNS]
    cat_idx = [X_train_cb.columns.get_loc(col) for col in cat_cols]
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)
    return X_train_cb, X_test_cb, cat_idx


def build_pipeline_specs(seed: int, n_jobs: int):
    numeric_features = None
    categorical_features = None

    def preprocessor_for(cols):
        nonlocal numeric_features, categorical_features
        numeric_features, categorical_features = split_feature_columns(cols)
        return build_preprocessor(numeric_features, categorical_features)

    specs = []
    for name, model in get_baseline_models(random_state=seed).items():
        if hasattr(model, "n_jobs"):
            try:
                model.set_params(n_jobs=n_jobs)
            except ValueError:
                pass
        specs.append((name, "baseline", model, False, "src.baselines.get_baseline_models"))

    ridge_params = strip_model_prefix(load_json(BASELINE_DIR / "ridge_classifier_best.json")["best_params"])
    specs.append(
        (
            "ridge_classifier_tuned",
            "baseline",
            RidgeClassifier(**ridge_params),
            False,
            "models/metrics/baseline/ridge_classifier_best.json",
        )
    )

    bagging_best = load_json(BAGGING_DIR / "bagging_tree_best_params.json")["models"]
    for name in ["bagging_decision_tree", "pasting_decision_tree"]:
        model_params, estimator_params = split_nested_estimator_params(bagging_best[name]["best_params"])
        estimator = DecisionTreeClassifier(random_state=seed, class_weight="balanced", **estimator_params)
        specs.append(
            (
                name,
                "bagging",
                BaggingClassifier(
                    estimator=estimator,
                    bootstrap=(name == "bagging_decision_tree"),
                    random_state=seed,
                    n_jobs=n_jobs,
                    **model_params,
                ),
                False,
                "models/metrics/bagging/bagging_tree_best_params.json",
            )
        )
    extra_params = strip_model_prefix(bagging_best["extra_trees"]["best_params"])
    specs.append(
        (
            "extra_trees_bagging_tuned",
            "bagging",
            ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs, **extra_params),
            False,
            "models/metrics/bagging/bagging_tree_best_params.json",
        )
    )

    classic = load_json(BOOST_DIR / "classic_boosting_best_params.json")["models"]
    ada_params, ada_estimator_params = split_nested_estimator_params(classic["adaboost_stumps"]["best_params"])
    specs.append(
        (
            "adaboost_stumps",
            "boost",
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(random_state=seed, **ada_estimator_params),
                random_state=seed,
                **ada_params,
            ),
            True,
            "models/metrics/boost/classic_boosting_best_params.json",
        )
    )
    gbdt_params = strip_model_prefix(classic["gbdt_log_loss"]["best_params"])
    specs.append(
        (
            "gbdt_log_loss",
            "boost",
            GradientBoostingClassifier(loss="log_loss", random_state=seed, **gbdt_params),
            True,
            "models/metrics/boost/classic_boosting_best_params.json",
        )
    )
    hist_params = strip_model_prefix(classic["hist_gradient_boosting"]["best_params"])
    specs.append(
        (
            "hist_gradient_boosting",
            "boost",
            HistGradientBoostingClassifier(loss="log_loss", early_stopping=False, random_state=seed, **hist_params),
            False,
            "models/metrics/boost/classic_boosting_best_params.json",
        )
    )

    svm_best = load_json(BASELINE_DIR / "svm_hyperparameter_fe_refine_dense_drop_noise_best.json")
    svm_params = json.loads(svm_best["best_by_test_among_top_cv_candidates"]["params_json"])
    svm_params = strip_model_prefix(svm_params)
    specs.append(
        (
            "svm_rbf_dense_drop_noise_best",
            "baseline",
            SVC(probability=True, random_state=seed, **svm_params),
            False,
            "models/metrics/baseline/svm_hyperparameter_fe_refine_dense_drop_noise_best.json",
        )
    )

    def instantiate_pipeline_specs(columns):
        preprocessor = preprocessor_for(columns)
        instantiated = []
        for name, family, model, sample_weight, source in specs:
            if name == "svm_rbf_dense_drop_noise_best":
                num_cols, cat_cols = split_feature_columns([c for c in columns if c not in NOISE_DROP_COLUMNS])
                svm_preprocessor = make_svm_preprocessor("dense", num_cols, cat_cols, svd_components=64)
                instantiated.append((name, family, Pipeline([("preprocessor", svm_preprocessor), ("model", model)]), sample_weight, source))
            elif name == "hist_gradient_boosting":
                instantiated.append(
                    (
                        name,
                        family,
                        Pipeline([("preprocessor", machine_only.dense_preprocessor(numeric_features, categorical_features)), ("model", model)]),
                        sample_weight,
                        source,
                    )
                )
            else:
                instantiated.append((name, family, Pipeline([("preprocessor", preprocessor), ("model", model)]), sample_weight, source))
        return instantiated

    return instantiate_pipeline_specs


def evaluate_special_models(seed, n_jobs, X_train, y_train, X_test, y_test, suffix: str = "", include_catboost: bool = True):
    records = []

    if include_catboost:
        X_train_cb, X_test_cb, cat_idx = prepare_catboost(X_train, X_test)
        catboost_params = {
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1:average=Macro",
            "auto_class_weights": "Balanced",
            "verbose": 0,
            "random_seed": seed,
            "thread_count": n_jobs,
            "allow_writing_files": False,
            **load_json(BOOST_DIR / "catboost_tuned_best_params.json"),
        }
        catboost = CatBoostClassifier(**catboost_params)
        catboost.fit(X_train_cb, y_train, cat_features=cat_idx)
        records.append(
            evaluate_record(
                f"catboost_tuned{suffix}",
                "boost",
                seed,
                y_test,
                catboost.predict(X_test_cb).ravel(),
                align_prob(catboost.predict_proba(X_test_cb), catboost.classes_),
                "models/metrics/boost/catboost_tuned_best_params.json",
            )
        )

    X_train_native, X_test_native, lgb_cat_cols = prepare_native_category(X_train, X_test)
    hist_column = load_json(BOOST_DIR / "hist_column_sampling_best_params.json")
    lgb_hist_params = hist_column["models"]["lightgbm"]["params"]
    lgb_configs = [
        (f"lightgbm_tuned{suffix}", load_json(BOOST_DIR / "lightgbm_tuned_best_params.json"), "models/metrics/boost/lightgbm_tuned_best_params.json"),
        (
            f"lightgbm_hist_column_best{suffix}",
            {
                "num_leaves": 63,
                "learning_rate": 0.03,
                "n_estimators": 300,
                "min_child_samples": 40,
                "max_depth": -1,
                "subsample": 0.8,
                "max_bin": int(lgb_hist_params["max_bin"]),
                "colsample_bytree": float(lgb_hist_params["column_fraction"]),
            },
            "models/metrics/boost/hist_column_sampling_best_params.json",
        ),
        (
            f"lightgbm_quantile_off_tuned{suffix}",
            load_json(BOOST_DIR / "lightgbm_quantile_off_best_params.json")["best_params"],
            "models/metrics/boost/lightgbm_quantile_off_best_params.json",
        ),
    ]
    for name, params, source in lgb_configs:
        lgb = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
            n_jobs=n_jobs,
            **params,
        )
        lgb.fit(X_train_native, y_train, categorical_feature=lgb_cat_cols)
        records.append(
            evaluate_record(name, "boost", seed, y_test, lgb.predict(X_test_native), align_prob(lgb.predict_proba(X_test_native), lgb.classes_), source)
        )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    xgb_hist_params = hist_column["models"]["xgboost_hist"]["params"]
    xgb_hist = XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABELS),
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.9,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        max_bin=int(xgb_hist_params["max_bin"]),
        colsample_bytree=float(xgb_hist_params["column_fraction"]),
    )
    xgb_hist.fit(X_train_native, y_train_encoded)
    pred_hist = label_encoder.inverse_transform(xgb_hist.predict(X_test_native).astype(int))
    records.append(
        evaluate_record(
            f"xgboost_hist_best{suffix}",
            "boost",
            seed,
            y_test,
            pred_hist,
            align_prob(xgb_hist.predict_proba(X_test_native), label_encoder.classes_),
            "models/metrics/boost/hist_column_sampling_best_params.json",
        )
    )

    numeric_cols, cat_cols = split_feature_columns(X_train.columns)
    onehot = build_preprocessor(numeric_cols, cat_cols)
    X_train_onehot = onehot.fit_transform(X_train)
    X_test_onehot = onehot.transform(X_test)
    xgb_exact_params = load_json(BOOST_DIR / "xgboost_exact_onehot_best_params.json")["best_params"]
    xgb_exact = XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABELS),
        tree_method="exact",
        enable_categorical=False,
        eval_metric="mlogloss",
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=n_jobs,
        **xgb_exact_params,
    )
    xgb_exact.fit(X_train_onehot, y_train_encoded)
    pred_exact = label_encoder.inverse_transform(xgb_exact.predict(X_test_onehot).astype(int))
    records.append(
        evaluate_record(
            f"xgboost_exact_onehot_local_tuned{suffix}",
            "boost",
            seed,
            y_test,
            pred_exact,
            align_prob(xgb_exact.predict_proba(X_test_onehot), label_encoder.classes_),
            "models/metrics/boost/xgboost_exact_onehot_best_params.json",
        )
    )

    return records


def evaluate_boost_drop_noise(seed, n_jobs, X_train, y_train, X_test, y_test, include_catboost: bool):
    X_train_dn = X_train.drop(columns=NOISE_DROP_COLUMNS, errors="ignore")
    X_test_dn = X_test.drop(columns=NOISE_DROP_COLUMNS, errors="ignore")
    records = evaluate_special_models(
        seed,
        n_jobs,
        X_train_dn,
        y_train,
        X_test_dn,
        y_test,
        suffix="_drop_noise",
        include_catboost=include_catboost,
    )

    classic = load_json(BOOST_DIR / "classic_boosting_best_params.json")["models"]
    numeric, categorical = split_feature_columns(X_train_dn.columns)
    dense = machine_only.dense_preprocessor(numeric, categorical)
    specs = [
        (
            "gbdt_log_loss_drop_noise",
            GradientBoostingClassifier(
                loss="log_loss",
                random_state=seed,
                **strip_model_prefix(classic["gbdt_log_loss"]["best_params"]),
            ),
            True,
        ),
        (
            "hist_gradient_boosting_drop_noise",
            HistGradientBoostingClassifier(
                loss="log_loss",
                early_stopping=False,
                random_state=seed,
                **strip_model_prefix(classic["hist_gradient_boosting"]["best_params"]),
            ),
            False,
        ),
    ]
    for name, model, sample_weight in specs:
        pipeline = Pipeline([("preprocessor", dense), ("model", model)])
        records.append(
            pipeline_predict_record(
                name,
                "boost",
                seed,
                pipeline,
                X_train_dn,
                y_train,
                X_test_dn,
                y_test,
                "drop-noise variant from scripts/reporting/compile_all_models_cross_seed_summary.py",
                sample_weight,
            )
        )
    return records


def evaluate_moe_v2(seed, n_jobs, X_train, y_train, X_test, y_test, cv_splits):
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values

    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=seed,
    )
    member_names, oof_probas, test_probas, _ = moe_base.build_member_probabilities(
        train_df,
        test_df,
        cv,
        numeric_features,
        categorical_features,
        n_jobs,
        quick=True,
    )
    losses = moe_base.true_class_losses(oof_probas, y_train)
    y_router = moe_base.pseudo_labels_from_losses(losses, "raw_loss")
    train_router, test_router = moe_v2.make_router_features(X_train, X_test, oof_probas, test_probas, "raw")
    rows, _, _ = moe_v2.evaluate_router_variant(
        "v2_raw",
        "logistic_router",
        train_router,
        test_router,
        y_router,
        losses,
        oof_probas,
        test_probas,
        y_train,
        y_test,
        cv_splits,
        n_jobs,
        "none",
    )
    out = []
    for row in rows:
        if row.get("split") != "test":
            continue
        model_name = "supervised_error_router_moe_v2_raw_logistic_none_" + row["routing"]
        out.append(
            {
                "model": model_name,
                "family": "ensemble",
                "set": "Testing",
                "seed": seed,
                "source": "scripts/ensemble/search_supervised_error_router_moe_v2.py",
                **{k: v for k, v in row.items() if k in METRIC_MAP},
            }
        )
    return out


def summarize(seed_rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (family, set_name, model), group in seed_rows.groupby(["family", "set", "model"], dropna=False):
        out = {
            "Model": model,
            "Family": family,
            "Set": set_name,
            "n_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(x) for x in sorted(group["seed"].unique())),
            "best_seed_by_f1_macro": int(group.loc[group["f1_macro"].idxmax(), "seed"]),
            "best_f1_macro": group["f1_macro"].max(),
            "worst_seed_by_f1_macro": int(group.loc[group["f1_macro"].idxmin(), "seed"]),
            "worst_f1_macro": group["f1_macro"].min(),
            "Source": " | ".join(sorted(group["source"].astype(str).unique())),
        }
        for col, label in METRIC_MAP.items():
            if col not in group:
                continue
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue
            out[f"{label} mean"] = values.mean()
            out[f"{label} std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            out[f"{label} min"] = values.min()
            out[f"{label} max"] = values.max()
        records.append(out)
    summary = pd.DataFrame(records)
    return summary.sort_values(["Set", "F1 macro mean"], ascending=[True, False])


def read_existing_dl_seed_rows() -> pd.DataFrame:
    rows = []
    for rel, model in [
        ("models/metrics/transformer/ft_seed_stability.csv", "ft_transformer_seed_stability"),
        ("models/metrics/transformer/viewwise_gated_ft_seed_stability.csv", "viewwise_gated_ft_seed_stability"),
    ]:
        path = PROJECT_ROOT / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["model"] = model
        df["family"] = "transformer"
        df["set"] = "Testing"
        df["source"] = rel
        rows.append(df)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS), help="Comma-separated split seeds.")
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--moe-cv", type=int, default=5)
    parser.add_argument("--skip-baseline", action="store_true", help="Only run special tuned tree/SVM models.")
    parser.add_argument("--skip-moe", action="store_true", help="Skip supervised error-router MoE v2 cross-seed evaluation.")
    parser.add_argument("--skip-boost-drop-noise", action="store_true", help="Skip boost drop-noise verification variants.")
    parser.add_argument("--include-catboost-drop-noise", action="store_true", help="Include slower CatBoost drop-noise verification.")
    parser.add_argument("--append-existing", action="store_true", help="Merge new seed-level rows with the existing output before summarizing.")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    X, y = load_full_processed_data()
    all_records = []

    for seed in seeds:
        print(f"[SEED] {seed}", flush=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=seed,
            stratify=y,
        )

        if not args.skip_baseline:
            instantiate_specs = build_pipeline_specs(seed, args.n_jobs)
            for name, family, pipeline, sample_weight, source in instantiate_specs(X_train.columns):
                X_train_fit = X_train.drop(columns=NOISE_DROP_COLUMNS, errors="ignore") if name == "svm_rbf_dense_drop_noise_best" else X_train
                X_test_fit = X_test.drop(columns=NOISE_DROP_COLUMNS, errors="ignore") if name == "svm_rbf_dense_drop_noise_best" else X_test
                print(f"[FIT] seed={seed} {name}", flush=True)
                all_records.append(
                    pipeline_predict_record(name, family, seed, pipeline, X_train_fit, y_train, X_test_fit, y_test, source, sample_weight)
                )

        for record in evaluate_special_models(seed, args.n_jobs, X_train, y_train, X_test, y_test):
            print(f"[DONE] seed={seed} {record['model']}", flush=True)
            all_records.append(record)

        if not args.skip_boost_drop_noise:
            for record in evaluate_boost_drop_noise(
                seed,
                args.n_jobs,
                X_train,
                y_train,
                X_test,
                y_test,
                include_catboost=args.include_catboost_drop_noise,
            ):
                print(f"[DENOISE] seed={seed} {record['model']}", flush=True)
                all_records.append(record)

        if not args.skip_moe:
            print(f"[MOE-V2] seed={seed} v2_raw logistic_router none", flush=True)
            all_records.extend(evaluate_moe_v2(seed, args.n_jobs, X_train, y_train, X_test, y_test, args.moe_cv))

    seed_df = pd.DataFrame(all_records)
    dl_df = read_existing_dl_seed_rows()
    if not dl_df.empty:
        seed_df = pd.concat([seed_df, dl_df], ignore_index=True, sort=False)
    if args.append_existing and SEED_LEVEL_OUT.exists():
        existing = pd.read_csv(SEED_LEVEL_OUT)
        seed_df = pd.concat([existing, seed_df], ignore_index=True, sort=False)
        dedup_cols = [col for col in ["model", "family", "set", "seed", "source"] if col in seed_df.columns]
        seed_df = seed_df.drop_duplicates(subset=dedup_cols, keep="last")

    REPORTING_DIR.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(SEED_LEVEL_OUT, index=False)
    summary = summarize(seed_df)
    summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Wrote {SEED_LEVEL_OUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {SUMMARY_OUT.relative_to(PROJECT_ROOT)}")
    print(f"Summary rows: {len(summary)}")


if __name__ == "__main__":
    main()
