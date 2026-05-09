"""Search SVM hyperparameters on the processed tabular dataset.

The SVM input keeps the established dense/one-hot preprocessing:
numeric features are imputed + standardized, categorical features are imputed
and one-hot encoded. CV is optimized for macro-F1. The final top candidates are
refit with probability=True so AUC can be reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns


warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet.*",
    category=FutureWarning,
)

OUTPUT_DIR = METRICS_DIR / "baseline"

APPLICATION_ORDER_COL = "Application order"
HIGH_CARDINALITY_CATEGORICAL = {
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
}
NOISE_DROP_COLUMNS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (without evaluations)",
    "International",
    "Educational special needs",
]


class ApplicationOrderThermometer(BaseEstimator, TransformerMixin):
    """Encode application order as cumulative thresholds instead of a scalar."""

    def __init__(self, max_order: int = 9):
        self.max_order = max_order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float).reshape(-1, 1)
        thresholds = np.arange(1, self.max_order + 1, dtype=float).reshape(1, -1)
        return (values >= thresholds).astype(float)


def output_paths(mode: str, feature_set: str, output_tag: str | None, selector: str = "none"):
    tag = output_tag or f"{mode}_{feature_set}"
    if selector != "none" and output_tag is None:
        tag = f"{tag}_{selector}"
    return (
        OUTPUT_DIR / f"svm_hyperparameter_{tag}_grid.csv",
        OUTPUT_DIR / f"svm_hyperparameter_{tag}_test_metrics.csv",
        OUTPUT_DIR / f"svm_hyperparameter_{tag}_best.json",
    )


def resolve_feature_set(feature_set: str) -> tuple[str, bool]:
    if feature_set.endswith("_drop_noise"):
        return feature_set.removesuffix("_drop_noise"), True
    return feature_set, False


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return value


def make_dense_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def make_app_thermo_preprocessor(numeric_features, categorical_features):
    numeric_without_app = [col for col in numeric_features if col != APPLICATION_ORDER_COL]
    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_without_app,
        ),
        (
            "app_order",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("thermo", ApplicationOrderThermometer(max_order=9)),
                ]
            ),
            [APPLICATION_ORDER_COL],
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            categorical_features,
        ),
    ]
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def make_mixed_target_preprocessor(numeric_features, categorical_features, use_app_thermo: bool):
    numeric_used = [col for col in numeric_features if not (use_app_thermo and col == APPLICATION_ORDER_COL)]
    high_card_cols = [col for col in categorical_features if col in HIGH_CARDINALITY_CATEGORICAL]
    low_card_cols = [col for col in categorical_features if col not in HIGH_CARDINALITY_CATEGORICAL]

    transformers = [
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_used,
        ),
        (
            "low_card_ohe",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            low_card_cols,
        ),
        (
            "high_card_te",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "target_encoder",
                        TargetEncoder(
                            target_type="multiclass",
                            cv=5,
                            shuffle=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                    ("scaler", StandardScaler()),
                ]
            ),
            high_card_cols,
        ),
    ]
    if use_app_thermo:
        transformers.append(
            (
                "app_order",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("thermo", ApplicationOrderThermometer(max_order=9)),
                    ]
                ),
                [APPLICATION_ORDER_COL],
            )
        )
    return ColumnTransformer(transformers=transformers, sparse_threshold=0.0)


def make_ohe_svd_preprocessor(numeric_features, categorical_features, svd_components: int):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ("svd", TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat_svd", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def make_preprocessor(feature_set: str, numeric_features, categorical_features, svd_components: int):
    feature_set, _ = resolve_feature_set(feature_set)
    if feature_set == "dense":
        return make_dense_preprocessor(numeric_features, categorical_features)
    if feature_set == "app_thermo":
        return make_app_thermo_preprocessor(numeric_features, categorical_features)
    if feature_set == "mixed_te":
        return make_mixed_target_preprocessor(numeric_features, categorical_features, use_app_thermo=False)
    if feature_set == "mixed_te_app_thermo":
        return make_mixed_target_preprocessor(numeric_features, categorical_features, use_app_thermo=True)
    if feature_set == "ohe_svd":
        return make_ohe_svd_preprocessor(numeric_features, categorical_features, svd_components)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def make_param_grid(mode: str):
    class_weight_options = [
        None,
        "balanced",
        {"Dropout": 1.0, "Enrolled": 1.25, "Graduate": 1.0},
        {"Dropout": 1.0, "Enrolled": 1.5, "Graduate": 1.0},
        {"Dropout": 1.0, "Enrolled": 1.75, "Graduate": 1.0},
    ]

    if mode == "rbf_refine":
        return [
            {
                "model__kernel": ["rbf"],
                "model__C": [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 30.0, 50.0],
                "model__gamma": [0.004, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02, 0.03],
                "model__class_weight": class_weight_options,
            },
        ]

    if mode == "fe_refine":
        return [
            {
                "model__kernel": ["rbf"],
                "model__C": [10.0, 15.0, 20.0, 30.0],
                "model__gamma": [0.003, 0.004, 0.006, 0.008, 0.01],
                "model__class_weight": [
                    None,
                    {"Dropout": 1.0, "Enrolled": 1.25, "Graduate": 1.0},
                    {"Dropout": 1.0, "Enrolled": 1.5, "Graduate": 1.0},
                    {"Dropout": 1.0, "Enrolled": 1.75, "Graduate": 1.0},
                ],
            },
        ]

    if mode == "rbf_final_refine":
        return [
            {
                "model__kernel": ["rbf"],
                "model__C": [20.0, 25.0, 30.0, 35.0, 40.0, 50.0],
                "model__gamma": [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, 0.006],
                "model__class_weight": [
                    None,
                    {"Dropout": 1.0, "Enrolled": 1.1, "Graduate": 1.0},
                    {"Dropout": 1.0, "Enrolled": 1.2, "Graduate": 1.0},
                    {"Dropout": 1.0, "Enrolled": 1.25, "Graduate": 1.0},
                    {"Dropout": 1.0, "Enrolled": 1.35, "Graduate": 1.0},
                    {"Dropout": 1.0, "Enrolled": 1.5, "Graduate": 1.0},
                ],
            },
        ]

    if mode == "selectk_refine":
        return [
            {
                "selector__k": [80, 120, 160, 200, "all"],
                "model__kernel": ["rbf"],
                "model__C": [10.0, 15.0, 20.0, 30.0],
                "model__gamma": [0.003, 0.004, 0.006, 0.008],
                "model__class_weight": [
                    None,
                    {"Dropout": 1.0, "Enrolled": 1.1, "Graduate": 1.0},
                    {"Dropout": 1.0, "Enrolled": 1.25, "Graduate": 1.0},
                    {"Dropout": 1.0, "Enrolled": 1.5, "Graduate": 1.0},
                ],
            },
        ]

    if mode == "quick":
        return [
            {
                "model__kernel": ["linear"],
                "model__C": [0.3, 1.0, 3.0, 10.0],
                "model__class_weight": ["balanced"],
            },
            {
                "model__kernel": ["rbf"],
                "model__C": [1.0, 3.0, 10.0],
                "model__gamma": ["scale", 0.01, 0.03, 0.1],
                "model__class_weight": ["balanced"],
            },
            {
                "model__kernel": ["poly"],
                "model__C": [1.0, 3.0],
                "model__gamma": ["scale", 0.01],
                "model__degree": [2, 3],
                "model__coef0": [0.0, 1.0],
                "model__class_weight": ["balanced"],
            },
        ]

    if mode == "focused":
        return [
            {
                "model__kernel": ["linear"],
                "model__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                "model__class_weight": ["balanced", None],
            },
            {
                "model__kernel": ["rbf"],
                "model__C": [0.3, 1.0, 3.0, 10.0, 30.0],
                "model__gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
                "model__class_weight": ["balanced", None],
            },
            {
                "model__kernel": ["poly"],
                "model__C": [0.3, 1.0, 3.0, 10.0],
                "model__gamma": ["scale", 0.003, 0.01, 0.03],
                "model__degree": [2, 3],
                "model__coef0": [0.0, 1.0],
                "model__class_weight": ["balanced"],
            },
        ]

    return [
        {
            "model__kernel": ["linear"],
            "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
            "model__class_weight": ["balanced", None],
        },
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
            "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
            "model__class_weight": ["balanced", None],
        },
        {
            "model__kernel": ["poly"],
            "model__C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
            "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03],
            "model__degree": [2, 3, 4],
            "model__coef0": [0.0, 0.5, 1.0],
            "model__class_weight": ["balanced"],
        },
        {
            "model__kernel": ["sigmoid"],
            "model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
            "model__gamma": ["scale", 0.001, 0.003, 0.01, 0.03],
            "model__coef0": [0.0, 0.5, 1.0],
            "model__class_weight": ["balanced"],
        },
    ]


def class_weight_name(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, dict):
        enrolled = value.get("Enrolled", "na")
        dropout = value.get("Dropout", "na")
        graduate = value.get("Graduate", "na")
        return f"custom_d{dropout}_e{enrolled}_g{graduate}"
    return str(value)


def model_name_from_params(params: dict[str, Any]) -> str:
    kernel = params.get("model__kernel", "unknown")
    c_val = params.get("model__C", "na")
    gamma = params.get("model__gamma", "na")
    degree = params.get("model__degree", "na")
    coef0 = params.get("model__coef0", "na")
    class_weight = class_weight_name(params.get("model__class_weight"))
    return f"svm_{kernel}_C{c_val}_gamma{gamma}_degree{degree}_coef0{coef0}_cw{class_weight}"


def evaluate_candidate(params, base_pipeline, X_train, y_train, X_test, y_test):
    estimator = clone(base_pipeline)
    estimator.set_params(**params)
    estimator.set_params(model__probability=True)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)
    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = model_name_from_params(params)
    metrics["kernel"] = params.get("model__kernel")
    metrics["C"] = params.get("model__C")
    metrics["gamma"] = params.get("model__gamma")
    metrics["degree"] = params.get("model__degree")
    metrics["coef0"] = params.get("model__coef0")
    metrics["class_weight"] = params.get("model__class_weight")
    metrics["selector_k"] = params.get("selector__k")
    metrics["params_json"] = json.dumps(json_safe(params), sort_keys=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["quick", "focused", "full", "rbf_refine", "fe_refine", "rbf_final_refine", "selectk_refine"],
        default="focused",
    )
    parser.add_argument(
        "--feature-set",
        choices=[
            "dense",
            "app_thermo",
            "mixed_te",
            "mixed_te_app_thermo",
            "ohe_svd",
            "dense_drop_noise",
            "app_thermo_drop_noise",
            "mixed_te_drop_noise",
            "mixed_te_app_thermo_drop_noise",
        ],
        default="dense",
        help="SVM-specific feature representation to evaluate.",
    )
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--top-k-test", type=int, default=20)
    parser.add_argument("--svd-components", type=int, default=64)
    parser.add_argument("--output-tag", default=None, help="Optional suffix for output files.")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--selector",
        choices=["none", "f_classif", "mutual_info"],
        default="none",
        help="Optional fold-internal feature selection after preprocessing.",
    )
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    _, drop_noise = resolve_feature_set(args.feature_set)
    dropped_columns: list[str] = []
    if drop_noise:
        dropped_columns = [col for col in NOISE_DROP_COLUMNS if col in X_train.columns]
        X_train = X_train.drop(columns=dropped_columns)
        X_test = X_test.drop(columns=dropped_columns)
        print(f"[FEATURES] dropped noise-like columns: {dropped_columns}", flush=True)

    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    pipeline_steps = [
        (
            "preprocessor",
            make_preprocessor(
                args.feature_set,
                numeric_features,
                categorical_features,
                args.svd_components,
            ),
        ),
    ]
    if args.selector == "f_classif":
        pipeline_steps.append(("selector", SelectKBest(score_func=f_classif, k="all")))
    elif args.selector == "mutual_info":
        pipeline_steps.append(
            (
                "selector",
                SelectKBest(
                    score_func=lambda X, y: mutual_info_classif(
                        X,
                        y,
                        discrete_features=False,
                        random_state=RANDOM_STATE,
                    ),
                    k="all",
                ),
            )
        )
    pipeline_steps.append(
        (
            "model",
            SVC(
                probability=False,
                cache_size=4096,
                random_state=RANDOM_STATE,
                max_iter=-1,
            ),
        )
    )
    base_pipeline = Pipeline(steps=pipeline_steps)
    grid_path, test_metrics_path, best_json_path = output_paths(
        args.mode,
        args.feature_set,
        args.output_tag,
        args.selector,
    )

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        base_pipeline,
        param_grid=make_param_grid(args.mode),
        scoring="f1_macro",
        cv=cv,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        return_train_score=True,
        refit=False,
    )
    print(
        f"[SEARCH] mode={args.mode} feature_set={args.feature_set} "
        f"selector={args.selector} cv={args.cv} n_jobs={args.n_jobs}",
        flush=True,
    )
    search.fit(X_train, y_train)

    cv_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score").reset_index(drop=True)
    keep_cols = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "std_train_score",
        "mean_fit_time",
        "params",
    ]
    grid_df = cv_df[keep_cols].copy()
    grid_df.insert(0, "search_mode", args.mode)
    grid_df.insert(1, "cv_splits", args.cv)
    grid_df.insert(2, "feature_set", args.feature_set)
    grid_df["model"] = grid_df["params"].map(model_name_from_params)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_table(grid_df.to_dict("records"), grid_path)

    top_params = [row["params"] for _, row in cv_df.head(args.top_k_test).iterrows()]
    test_records = []
    for idx, params in enumerate(top_params, start=1):
        print(f"[EVAL] top_cv_rank={idx} params={params}", flush=True)
        metrics = evaluate_candidate(params, base_pipeline, X_train, y_train, X_test, y_test)
        metrics["search_mode"] = args.mode
        metrics["feature_set"] = args.feature_set
        metrics["selector"] = args.selector
        metrics["cv_splits"] = args.cv
        metrics["cv_rank"] = idx
        match = grid_df.iloc[idx - 1]
        metrics["cv_mean_f1_macro"] = float(match["mean_test_score"])
        metrics["cv_std_f1_macro"] = float(match["std_test_score"])
        test_records.append(metrics)

    test_df_metrics = pd.DataFrame(test_records).sort_values(
        ["f1_macro", "roc_auc_ovr_macro", "accuracy"],
        ascending=[False, False, False],
    )
    save_metrics_table(test_df_metrics.to_dict("records"), test_metrics_path)

    best_by_test = test_df_metrics.iloc[0].to_dict()
    best_by_cv = test_df_metrics.sort_values("cv_rank").iloc[0].to_dict()
    best_json_path.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "feature_set": args.feature_set,
                "selector": args.selector,
                "dropped_columns": dropped_columns,
                "cv_splits": args.cv,
                "n_jobs": args.n_jobs,
                "svd_components": args.svd_components,
                "selection_metric": "CV macro-F1; top candidates also evaluated on held-out test",
                "best_by_cv": json_safe(best_by_cv),
                "best_by_test_among_top_cv_candidates": json_safe(best_by_test),
                "grid_path": str(grid_path.relative_to(PROJECT_ROOT)),
                "test_metrics_path": str(test_metrics_path.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[TOP TEST CANDIDATES]")
    print(
        test_df_metrics[
            [
                "model",
                "cv_rank",
                "cv_mean_f1_macro",
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
                "dropout_precision",
                "enrolled_precision",
                "graduate_precision",
            ]
        ].head(10).to_string(index=False)
    )
    print(f"saved grid: {grid_path}")
    print(f"saved test metrics: {test_metrics_path}")
    print(f"saved best json: {best_json_path}")


if __name__ == "__main__":
    main()
