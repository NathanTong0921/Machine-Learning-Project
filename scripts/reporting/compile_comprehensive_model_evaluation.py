"""Create report-style train/test/entire metrics for reproducible ML models.

The output format mirrors the course-style table:
Model, Set, Accuracy, weighted precision, weighted recall, weighted F1, and
macro F1.

The models are fit only on `data/processed/train.csv`. The `Entire` row is the
combined score on train + test predictions from that train-fitted model.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

if not hasattr(np, "NaN"):
    np.NaN = np.nan

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import get_baseline_models
from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor, split_feature_columns


BOOST_METRICS_DIR = METRICS_DIR / "boost"
REPORTING_DIR = METRICS_DIR / "reporting"
OUTPUT_CSV = REPORTING_DIR / "comprehensive_model_evaluation.csv"
OUTPUT_MD = REPORTING_DIR / "comprehensive_model_evaluation.md"
COURSE_STYLE_OUTPUT_CSV = REPORTING_DIR / "comprehensive_model_evaluation_course_style.csv"
COURSE_STYLE_OUTPUT_MD = REPORTING_DIR / "comprehensive_model_evaluation_course_style.md"
DETAILED_OUTPUT_CSV = REPORTING_DIR / "comprehensive_model_evaluation_detailed.csv"
DETAILED_OUTPUT_MD = REPORTING_DIR / "comprehensive_model_evaluation_detailed.md"
ALL_RESULTS_CSV = REPORTING_DIR / "all_model_results_detailed.csv"
ALL_RESULTS_MD = REPORTING_DIR / "all_model_results_detailed.md"

SUMMARY_COLUMNS = [
    "Model",
    "Family",
    "Set",
    "Accuracy",
    "Prec (w)",
    "Rec (w)",
    "F1 (w)",
    "F1 (macro)",
    "AUC",
    "FMI",
    "Dropout precision",
    "Dropout recall",
    "Dropout F1",
    "Dropout acc",
    "Enrolled precision",
    "Enrolled recall",
    "Enrolled F1",
    "Enrolled acc",
    "Graduate precision",
    "Graduate recall",
    "Graduate F1",
    "Graduate acc",
    "Source",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_data():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    return X_train, y_train, X_test, y_test


def table_row(model_name: str, set_name: str, y_true, y_pred):
    metrics = classification_metrics(y_true, y_pred, labels=LABELS)
    return {
        "Model": model_name,
        "Family": "retrained_ml",
        "Set": set_name,
        "Accuracy": metrics["accuracy"],
        "Prec (w)": metrics["precision_weighted"],
        "Rec (w)": metrics["recall_weighted"],
        "F1 (w)": metrics["f1_weighted"],
        "F1 (macro)": metrics["f1_macro"],
        "AUC": np.nan,
        "FMI": metrics["fmi"],
        "Dropout precision": metrics["dropout_precision"],
        "Dropout recall": metrics["dropout_recall"],
        "Dropout F1": metrics["dropout_f1"],
        "Dropout acc": metrics["dropout_accuracy"],
        "Enrolled precision": metrics["enrolled_precision"],
        "Enrolled recall": metrics["enrolled_recall"],
        "Enrolled F1": metrics["enrolled_f1"],
        "Enrolled acc": metrics["enrolled_accuracy"],
        "Graduate precision": metrics["graduate_precision"],
        "Graduate recall": metrics["graduate_recall"],
        "Graduate F1": metrics["graduate_f1"],
        "Graduate acc": metrics["graduate_accuracy"],
        "Source": "scripts/reporting/compile_comprehensive_model_evaluation.py",
    }


def evaluate_predictions(model_name: str, y_train, pred_train, y_test, pred_test):
    y_entire = pd.concat([pd.Series(y_train), pd.Series(y_test)], ignore_index=True)
    pred_entire = pd.Series(list(pred_train) + list(pred_test))
    return [
        table_row(model_name, "Training", y_train, pred_train),
        table_row(model_name, "Testing", y_test, pred_test),
        table_row(model_name, "Entire", y_entire, pred_entire),
    ]


def fit_predict_pipeline_models(X_train, y_train, X_test, y_test):
    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    rows = []

    for model_name, model in get_baseline_models(random_state=RANDOM_STATE).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", model),
            ]
        )
        print(f"[FIT] {model_name}", flush=True)
        pipeline.fit(X_train, y_train)
        rows.extend(
            evaluate_predictions(
                model_name,
                y_train,
                pipeline.predict(X_train),
                y_test,
                pipeline.predict(X_test),
            )
        )
    return rows


def prepare_catboost_data(X_train, X_test):
    X_train_cb = X_train.copy()
    X_test_cb = X_test.copy()
    cat_cols = [col for col in X_train_cb.columns if col in CATEGORICAL_COLUMNS]
    cat_idx = [X_train_cb.columns.get_loc(col) for col in cat_cols]
    for col in cat_cols:
        X_train_cb[col] = X_train_cb[col].astype(str)
        X_test_cb[col] = X_test_cb[col].astype(str)
    return X_train_cb, X_test_cb, cat_idx


def prepare_native_category_data(X_train, X_test):
    X_train_native = X_train.copy()
    X_test_native = X_test.copy()
    cat_cols = [col for col in X_train_native.columns if col in CATEGORICAL_COLUMNS]
    for col in cat_cols:
        X_train_native[col] = X_train_native[col].astype("category")
        X_test_native[col] = X_test_native[col].astype("category")
    return X_train_native, X_test_native, cat_cols


def fit_predict_catboost_models(X_train, y_train, X_test, y_test):
    X_train_cb, X_test_cb, cat_idx = prepare_catboost_data(X_train, X_test)
    configs = [
        (
            "catboost",
            {
                "loss_function": "MultiClass",
                "eval_metric": "TotalF1:average=Macro",
                "auto_class_weights": "Balanced",
                "depth": 6,
                "learning_rate": 0.03,
                "iterations": 500,
                "verbose": 0,
                "random_seed": RANDOM_STATE,
                "allow_writing_files": False,
            },
        ),
        (
            "catboost_tuned",
            {
                "loss_function": "MultiClass",
                "eval_metric": "TotalF1:average=Macro",
                "auto_class_weights": "Balanced",
                "verbose": 0,
                "random_seed": RANDOM_STATE,
                "thread_count": -1,
                "allow_writing_files": False,
                **load_json(BOOST_METRICS_DIR / "catboost_tuned_best_params.json"),
            },
        ),
    ]

    rows = []
    for model_name, params in configs:
        print(f"[FIT] {model_name}", flush=True)
        model = CatBoostClassifier(**params)
        model.fit(X_train_cb, y_train, cat_features=cat_idx)
        rows.extend(
            evaluate_predictions(
                model_name,
                y_train,
                model.predict(X_train_cb).ravel(),
                y_test,
                model.predict(X_test_cb).ravel(),
            )
        )
    return rows


def fit_predict_lightgbm_models(X_train, y_train, X_test, y_test):
    X_train_lgb, X_test_lgb, cat_cols = prepare_native_category_data(X_train, X_test)
    tuned_params = load_json(BOOST_METRICS_DIR / "lightgbm_tuned_best_params.json")
    hist_params = load_json(BOOST_METRICS_DIR / "hist_column_sampling_best_params.json")["models"]["lightgbm"]["params"]

    configs = [
        (
            "lightgbm_tuned",
            {
                "objective": "multiclass",
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "verbosity": -1,
                "n_jobs": 16,
                **tuned_params,
            },
        ),
        (
            "lightgbm_hist_column_best",
            {
                "objective": "multiclass",
                "class_weight": "balanced",
                "num_leaves": 63,
                "learning_rate": 0.03,
                "n_estimators": 300,
                "min_child_samples": 40,
                "max_depth": -1,
                "subsample": 0.8,
                "random_state": RANDOM_STATE,
                "verbosity": -1,
                "n_jobs": 16,
                "max_bin": int(hist_params["max_bin"]),
                "colsample_bytree": float(hist_params["column_fraction"]),
            },
        ),
    ]

    rows = []
    for model_name, params in configs:
        print(f"[FIT] {model_name}", flush=True)
        model = LGBMClassifier(**params)
        model.fit(X_train_lgb, y_train, categorical_feature=cat_cols)
        rows.extend(
            evaluate_predictions(
                model_name,
                y_train,
                model.predict(X_train_lgb),
                y_test,
                model.predict(X_test_lgb),
            )
        )
    return rows


def fit_predict_xgboost_models(X_train, y_train, X_test, y_test):
    rows = []

    X_train_hist, X_test_hist, _ = prepare_native_category_data(X_train, X_test)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    hist_params = load_json(BOOST_METRICS_DIR / "hist_column_sampling_best_params.json")["models"]["xgboost_hist"]["params"]
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
        random_state=RANDOM_STATE,
        n_jobs=16,
        max_bin=int(hist_params["max_bin"]),
        colsample_bytree=float(hist_params["column_fraction"]),
    )
    print("[FIT] xgboost_hist_best", flush=True)
    xgb_hist.fit(X_train_hist, y_train_encoded)
    rows.extend(
        evaluate_predictions(
            "xgboost_hist_best",
            y_train,
            label_encoder.inverse_transform(xgb_hist.predict(X_train_hist).astype(int)),
            y_test,
            label_encoder.inverse_transform(xgb_hist.predict(X_test_hist).astype(int)),
        )
    )

    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_exact = preprocessor.fit_transform(X_train)
    X_test_exact = preprocessor.transform(X_test)
    xgb_exact = XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABELS),
        tree_method="exact",
        enable_categorical=False,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=16,
    )
    print("[FIT] xgboost_exact_onehot", flush=True)
    xgb_exact.fit(X_train_exact, y_train_encoded)
    rows.extend(
        evaluate_predictions(
            "xgboost_exact_onehot",
            y_train,
            label_encoder.inverse_transform(xgb_exact.predict(X_train_exact).astype(int)),
            y_test,
            label_encoder.inverse_transform(xgb_exact.predict(X_test_exact).astype(int)),
        )
    )
    return rows


def write_markdown(df: pd.DataFrame, output_path: Path, metric_cols: list[str] | None = None):
    rounded = df.copy()
    if metric_cols is None:
        metric_cols = [
            "Accuracy",
            "Prec (w)",
            "Rec (w)",
            "F1 (w)",
            "F1 (macro)",
            "AUC",
            "FMI",
            "Dropout precision",
            "Dropout recall",
            "Dropout F1",
            "Dropout acc",
            "Enrolled precision",
            "Enrolled recall",
            "Enrolled F1",
            "Enrolled acc",
            "Graduate precision",
            "Graduate recall",
            "Graduate F1",
            "Graduate acc",
        ]
    for col in metric_cols:
        if col in rounded.columns:
            rounded[col] = rounded[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    output_path.write_text(rounded.to_markdown(index=False), encoding="utf-8")


def infer_family(path: Path) -> str:
    rel = path.relative_to(METRICS_DIR)
    return rel.parts[0] if len(rel.parts) > 1 else "root"


def infer_set(row: pd.Series) -> str:
    if "Set" in row and pd.notna(row["Set"]):
        return str(row["Set"])
    if "split" in row and pd.notna(row["split"]):
        split = str(row["split"]).lower()
        if split == "test":
            return "Testing"
        if split == "oof":
            return "OOF"
        return split.upper()
    model = str(row.get("model", "")).lower()
    if model.endswith("_test") or "_test_" in model:
        return "Testing"
    if model.endswith("_oof") or "_oof_" in model:
        return "OOF"
    if "cv" in model or "fold" in model:
        return "CV"
    return "Testing"


def first_value(row: pd.Series, names: list[str]):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return np.nan


def normalize_metric_row(row: pd.Series, source: Path) -> dict[str, Any]:
    rel_source = source.relative_to(PROJECT_ROOT)
    model = str(row["model"]) if "model" in row and pd.notna(row["model"]) else source.stem
    return {
        "Model": model,
        "Family": infer_family(source),
        "Set": infer_set(row),
        "Accuracy": first_value(row, ["Accuracy", "accuracy", "test_accuracy", "oof_accuracy"]),
        "Prec (w)": first_value(row, ["Prec (w)", "precision_weighted", "weighted_precision"]),
        "Rec (w)": first_value(row, ["Rec (w)", "recall_weighted", "weighted_recall"]),
        "F1 (w)": first_value(row, ["F1 (w)", "f1_weighted", "test_f1_weighted", "oof_f1_weighted"]),
        "F1 (macro)": first_value(row, ["F1 (macro)", "f1_macro", "test_f1_macro", "oof_f1_macro"]),
        "AUC": first_value(row, ["AUC", "roc_auc_ovr_macro", "test_roc_auc_ovr_macro", "oof_roc_auc_ovr_macro"]),
        "FMI": first_value(row, ["FMI", "fmi"]),
        "Dropout precision": first_value(row, ["Dropout precision", "dropout_precision"]),
        "Dropout recall": first_value(row, ["Dropout recall", "dropout_recall", "dropout_accuracy"]),
        "Dropout F1": first_value(row, ["Dropout F1", "dropout_f1"]),
        "Dropout acc": first_value(row, ["Dropout acc", "dropout_accuracy", "dropout_recall"]),
        "Enrolled precision": first_value(row, ["Enrolled precision", "enrolled_precision"]),
        "Enrolled recall": first_value(row, ["Enrolled recall", "enrolled_recall", "enrolled_accuracy"]),
        "Enrolled F1": first_value(row, ["Enrolled F1", "enrolled_f1"]),
        "Enrolled acc": first_value(row, ["Enrolled acc", "enrolled_accuracy", "enrolled_recall"]),
        "Graduate precision": first_value(row, ["Graduate precision", "graduate_precision"]),
        "Graduate recall": first_value(row, ["Graduate recall", "graduate_recall", "graduate_accuracy"]),
        "Graduate F1": first_value(row, ["Graduate F1", "graduate_f1"]),
        "Graduate acc": first_value(row, ["Graduate acc", "graduate_accuracy", "graduate_recall"]),
        "Source": str(rel_source),
    }


def has_metric_columns(df: pd.DataFrame) -> bool:
    metric_cols = {
        "Accuracy",
        "accuracy",
        "test_accuracy",
        "oof_accuracy",
        "F1 (macro)",
        "f1_macro",
        "test_f1_macro",
        "oof_f1_macro",
        "roc_auc_ovr_macro",
    }
    return bool(metric_cols.intersection(df.columns))


def is_final_metric_file(path: Path) -> bool:
    stem = path.stem
    skip_tokens = [
        "grid",
        "search",
        "cv_results",
        "training_history",
        "feature_importance",
        "gate_summary",
        "cluster_summary",
        "member_oof_scores",
        "weights",
        "ranked",
    ]
    return not any(token in stem for token in skip_tokens)


def collect_historical_metric_rows() -> pd.DataFrame:
    excluded = {
        OUTPUT_CSV.name,
        COURSE_STYLE_OUTPUT_CSV.name,
        DETAILED_OUTPUT_CSV.name,
        ALL_RESULTS_CSV.name,
    }
    rows = []
    for path in sorted(METRICS_DIR.rglob("*.csv")):
        if path.parent == REPORTING_DIR and path.name in excluded:
            continue
        if path.parent == METRICS_DIR:
            continue
        if not is_final_metric_file(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[SKIP] cannot read {path}: {exc}", flush=True)
            continue
        if df.empty or not has_metric_columns(df):
            continue
        for _, row in df.iterrows():
            normalized = normalize_metric_row(row, path)
            if pd.notna(normalized["Accuracy"]) or pd.notna(normalized["F1 (macro)"]):
                rows.append(normalized)
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    order = {"Training": 0, "Testing": 1, "Entire": 2, "OOF": 3, "CV": 4}
    df = df.copy()
    df["_set_order"] = df["Set"].map(order).fillna(9)
    return df.sort_values(["Family", "Model", "_set_order", "Source"]).drop(columns=["_set_order"])


def main():
    X_train, y_train, X_test, y_test = load_data()
    REPORTING_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(fit_predict_pipeline_models(X_train, y_train, X_test, y_test))
    rows.extend(fit_predict_catboost_models(X_train, y_train, X_test, y_test))
    rows.extend(fit_predict_lightgbm_models(X_train, y_train, X_test, y_test))
    rows.extend(fit_predict_xgboost_models(X_train, y_train, X_test, y_test))

    df = pd.DataFrame(rows)
    order = {"Training": 0, "Testing": 1, "Entire": 2}
    df["_set_order"] = df["Set"].map(order)
    df = df.sort_values(["Model", "_set_order"]).drop(columns=["_set_order"])

    compact_cols = ["Model", "Set", "Accuracy", "Prec (w)", "Rec (w)", "F1 (w)", "F1 (macro)"]
    save_metrics_table(df[compact_cols].to_dict("records"), COURSE_STYLE_OUTPUT_CSV)
    write_markdown(
        df[compact_cols],
        COURSE_STYLE_OUTPUT_MD,
        metric_cols=["Accuracy", "Prec (w)", "Rec (w)", "F1 (w)", "F1 (macro)"],
    )
    save_metrics_table(df[SUMMARY_COLUMNS].to_dict("records"), OUTPUT_CSV)
    write_markdown(df[SUMMARY_COLUMNS], OUTPUT_MD)
    save_metrics_table(df[SUMMARY_COLUMNS].to_dict("records"), DETAILED_OUTPUT_CSV)
    write_markdown(df[SUMMARY_COLUMNS], DETAILED_OUTPUT_MD)

    historical_df = collect_historical_metric_rows()
    all_results_df = pd.concat([df[SUMMARY_COLUMNS], historical_df], ignore_index=True)
    all_results_df = sort_summary(all_results_df)
    save_metrics_table(all_results_df.to_dict("records"), ALL_RESULTS_CSV)
    write_markdown(all_results_df, ALL_RESULTS_MD)

    print(df.to_string(index=False))
    print(f"saved csv: {OUTPUT_CSV}")
    print(f"saved markdown: {OUTPUT_MD}")
    print(f"saved course-style csv: {COURSE_STYLE_OUTPUT_CSV}")
    print(f"saved course-style markdown: {COURSE_STYLE_OUTPUT_MD}")
    print(f"saved detailed csv: {DETAILED_OUTPUT_CSV}")
    print(f"saved detailed markdown: {DETAILED_OUTPUT_MD}")
    print(f"saved all-results csv: {ALL_RESULTS_CSV}")
    print(f"saved all-results markdown: {ALL_RESULTS_MD}")


if __name__ == "__main__":
    main()
