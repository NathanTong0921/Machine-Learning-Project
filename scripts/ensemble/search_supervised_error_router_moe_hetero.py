"""Test heterogeneous expert pools for the supervised error-router MoE.

This tests whether replacing the third Core3 tree expert with a linear or SVM
expert gives the router a more orthogonal basis:

- CatBoost tuned
- LightGBM hist-column best
- one of: GBDT control, PCA-fused thermometer ElasticNet Logistic, PCA-fused
  thermometer calibrated LinearSVC
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import search_supervised_error_router_moe as base
import search_supervised_error_router_moe_v2 as v2
import tune_machine_only_ensemble as machine_only

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import save_metrics_table
from src.preprocessing import split_feature_columns


CURRICULAR_PCA_COLUMNS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]


def load_tree_params():
    classic = machine_only.load_json(METRICS_DIR / "classic_boosting_best_params.json")["models"]
    hist = machine_only.load_json(METRICS_DIR / "hist_column_sampling_best_params.json")["models"]
    catboost_params = machine_only.load_json(METRICS_DIR / "catboost_tuned_best_params.json")
    lightgbm_params = machine_only.load_json(METRICS_DIR / "lightgbm_tuned_best_params.json")
    lightgbm_hist_params = {
        **lightgbm_params,
        "max_bin": hist["lightgbm"]["params"]["max_bin"],
        "colsample_bytree": hist["lightgbm"]["params"]["column_fraction"],
    }
    return catboost_params, lightgbm_hist_params, classic["gbdt_log_loss"]["best_params"]


def make_thermo_frame(df):
    features = df.drop(columns=[TARGET_COL]).copy()
    features = v2.add_router_thermometer(features)
    out = features.copy()
    out[TARGET_COL] = df[TARGET_COL].values
    return out


def make_dense_linear_model(name, seed):
    if name == "elasticnet_logreg_c05_l02":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.2,
            C=0.5,
            max_iter=8000,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    elif name == "elasticnet_logreg_c02_l05":
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.2,
            max_iter=8000,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    elif name == "linear_svc_c05":
        model = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=0.5,
                class_weight="balanced",
                dual="auto",
                max_iter=8000,
                random_state=seed,
            ),
            method="sigmoid",
            cv=3,
        )
    elif name == "linear_svc_c1":
        model = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=8000,
                random_state=seed,
            ),
            method="sigmoid",
            cv=3,
        )
    else:
        raise ValueError(f"Unknown dense linear/SVM expert: {name}")
    return model


def pca_fused_dense_preprocessor(columns):
    numeric_features, categorical_features = split_feature_columns(columns)
    pca_features = [col for col in CURRICULAR_PCA_COLUMNS if col in numeric_features]
    numeric_remainder = [col for col in numeric_features if col not in pca_features]
    transformers = []
    if pca_features:
        transformers.append(
            (
                "curricular_pca",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=min(2, len(pca_features)), random_state=RANDOM_STATE)),
                    ]
                ),
                pca_features,
            )
        )
    if numeric_remainder:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_remainder,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def dense_expert_oof_test(name, train_df, test_df, cv):
    train_aug = make_thermo_frame(train_df)
    test_aug = make_thermo_frame(test_df)
    X = train_aug.drop(columns=[TARGET_COL])
    y = train_aug[TARGET_COL]
    X_test = test_aug.drop(columns=[TARGET_COL])
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[HETERO] member={name} fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        model = Pipeline(
            steps=[
                ("preprocessor", pca_fused_dense_preprocessor(X.columns)),
                ("model", make_dense_linear_model(name, RANDOM_STATE + fold_idx)),
            ]
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx], model__sample_weight=sample_weight[train_idx])
        oof[valid_idx] = machine_only.align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)
    model = Pipeline(
        steps=[
            ("preprocessor", pca_fused_dense_preprocessor(X.columns)),
            ("model", make_dense_linear_model(name, RANDOM_STATE)),
        ]
    )
    model.fit(X, y, model__sample_weight=sample_weight)
    test_proba = machine_only.align_proba(model.predict_proba(X_test), model.classes_)
    return oof, test_proba


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--output-prefix", default="supervised_error_router_moe_hetero")
    parser.add_argument("--skip-linear-svc", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    catboost_params, lightgbm_hist_params, gbdt_params = load_tree_params()

    print("[HETERO] building shared CatBoost and LightGBM hist OOF/test", flush=True)
    cat_oof, cat_test = machine_only.catboost_oof_test(catboost_params, train_df, test_df, cv)
    lgb_oof, lgb_test = machine_only.lightgbm_oof_test("lightgbm_hist_column_best", lightgbm_hist_params, train_df, test_df, cv, args.n_jobs)
    gbdt_oof, gbdt_test = machine_only.sklearn_oof_test(
        "gbdt_log_loss",
        gbdt_params,
        train_df,
        test_df,
        cv,
        numeric_features,
        categorical_features,
    )

    candidates = [
        ("gbdt_log_loss_control", gbdt_oof, gbdt_test),
        ("elasticnet_logreg_c05_l02", *dense_expert_oof_test("elasticnet_logreg_c05_l02", train_df, test_df, cv)),
        ("elasticnet_logreg_c02_l05", *dense_expert_oof_test("elasticnet_logreg_c02_l05", train_df, test_df, cv)),
    ]
    if not args.skip_linear_svc:
        candidates.extend([
            ("linear_svc_c05", *dense_expert_oof_test("linear_svc_c05", train_df, test_df, cv)),
            ("linear_svc_c1", *dense_expert_oof_test("linear_svc_c1", train_df, test_df, cv)),
        ])

    records = []
    for third_name, third_oof, third_test in candidates:
        member_names = ["catboost_tuned", "lightgbm_hist_column_best", third_name]
        oof_probas = [cat_oof, lgb_oof, third_oof]
        test_probas = [cat_test, lgb_test, third_test]
        for member_name, test_proba in zip(member_names, test_probas):
            row = base.evaluate(f"hetero_member_{third_name}_{member_name}_test", y_test, test_proba)
            row.update({"member_set": third_name, "experiment": "hetero_member"})
            records.append(row)

        losses = base.true_class_losses(oof_probas, y_train)
        y_router = base.pseudo_labels_from_losses(losses, "raw_loss")
        pseudo_counts = base.assignment_counts_payload(y_router, member_names)
        pseudo_shares = base.assignment_share_payload(y_router, member_names)
        for feature_variant in ["raw", "meta"]:
            train_router, test_router = v2.make_router_features(X_train, X_test, oof_probas, test_probas, feature_variant)
            for weight_mode in ["none", "sqrt_margin"]:
                print(f"[HETERO] set={third_name} features={feature_variant} weight={weight_mode}", flush=True)
                rows, _, _ = v2.evaluate_router_variant(
                    f"hetero_{third_name}_{feature_variant}",
                    "logistic_router",
                    train_router,
                    test_router,
                    y_router,
                    losses,
                    oof_probas,
                    test_probas,
                    y_train,
                    y_test,
                    args.cv,
                    args.n_jobs,
                    weight_mode,
                )
                for row in rows:
                    row.update({
                        "member_set": third_name,
                        "router_feature_variant": feature_variant,
                        "members_json": json.dumps(member_names, ensure_ascii=False),
                        "pseudo_label_counts_json": json.dumps(pseudo_counts, ensure_ascii=False, sort_keys=True),
                        "pseudo_label_shares_json": json.dumps(pseudo_shares, ensure_ascii=False, sort_keys=True),
                    })
                records.extend(rows)
                save_metrics_table(records, METRICS_DIR / f"{args.output_prefix}_metrics.csv")

    metrics_df = pd.DataFrame(records)
    metrics_path = METRICS_DIR / f"{args.output_prefix}_metrics.csv"
    best_df = metrics_df[metrics_df["model"].str.contains("_test", na=False)].sort_values("f1_macro", ascending=False)
    best = best_df.iloc[0].to_dict()
    save_metrics_table(metrics_df.to_dict(orient="records"), metrics_path)
    (METRICS_DIR / f"{args.output_prefix}_best.json").write_text(
        json.dumps(v2.json_safe(best), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(best_df[[
        "model",
        "member_set",
        "accuracy",
        "f1_macro",
        "roc_auc_ovr_macro",
        "fmi",
        "dropout_accuracy",
        "enrolled_accuracy",
        "graduate_accuracy",
    ]].head(12).to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved best: {METRICS_DIR / f'{args.output_prefix}_best.json'}")


if __name__ == "__main__":
    main()
