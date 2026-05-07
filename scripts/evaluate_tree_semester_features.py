"""Evaluate tuned tree models with semester ratio/delta features.

Declaration:
- This script reuses existing tuned CatBoost/LightGBM parameters.
- It adds deterministic semester ratio/delta features with no target leakage.
- It reports 5-fold OOF metrics and a final held-out test evaluation.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_metrics_table_figure


CV_SPLITS = 5
LIGHTGBM_N_JOBS = 108


def safe_divide(numerator, denominator):
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def add_semester_ratio_features(df):
    df = df.copy()
    first_enrolled = df["Curricular units 1st sem (enrolled)"]
    second_enrolled = df["Curricular units 2nd sem (enrolled)"]
    first_evaluations = df["Curricular units 1st sem (evaluations)"]
    second_evaluations = df["Curricular units 2nd sem (evaluations)"]
    first_approved = df["Curricular units 1st sem (approved)"]
    second_approved = df["Curricular units 2nd sem (approved)"]

    df["first_approved_rate"] = safe_divide(first_approved, first_enrolled)
    df["second_approved_rate"] = safe_divide(second_approved, second_enrolled)
    df["first_eval_pass_rate"] = safe_divide(first_approved, first_evaluations)
    df["second_eval_pass_rate"] = safe_divide(second_approved, second_evaluations)
    df["grade_delta"] = df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    df["approved_delta"] = second_approved - first_approved
    df["tuition_scholarship_interaction"] = df["Tuition fees up to date"] * df["Scholarship holder"]
    df["debtor_tuition_interaction"] = df["Debtor"] * df["Tuition fees up to date"]
    return df


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_proba(proba, classes):
    aligned = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    class_to_col = {str(label): idx for idx, label in enumerate(classes)}
    for out_idx, label in enumerate(LABELS):
        aligned[:, out_idx] = proba[:, class_to_col[label]]
    return aligned


def evaluate(name, y_true, pred, proba=None):
    metrics = classification_metrics(y_true, pred, y_prob=proba, labels=LABELS if proba is not None else None)
    report = classification_report(y_true, pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["model"] = name
    for label in LABELS:
        metrics[f"{label.lower()}_precision"] = report[label]["precision"]
        metrics[f"{label.lower()}_recall"] = report[label]["recall"]
        metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]
    return metrics


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


def make_catboost(params, seed):
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=seed,
        thread_count=-1,
        **params,
    )


def make_lightgbm(params, seed):
    return LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
        n_jobs=LIGHTGBM_N_JOBS,
        **params,
    )


def run_member(name, train_df, test_df, params, prepare_xy, build_model):
    X_train, y_train, categorical = prepare_xy(train_df)
    X_test, y_test, _ = prepare_xy(test_df)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_records = []
    oof_proba = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
        print(f"[PROGRESS] member={name} fold={fold_idx}/{CV_SPLITS}", flush=True)
        model = build_model(params, RANDOM_STATE + fold_idx)
        fit_kwargs = {"cat_features": categorical} if name.startswith("catboost") else {"categorical_feature": categorical}
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], **fit_kwargs)
        valid_proba = align_proba(model.predict_proba(X_train.iloc[valid_idx]), model.classes_)
        valid_pred = np.array([LABELS[idx] for idx in valid_proba.argmax(axis=1)])
        oof_proba[valid_idx] = valid_proba
        fold_metrics = evaluate(f"{name}_fold", y_train.iloc[valid_idx], valid_pred, proba=valid_proba)
        fold_metrics["fold"] = fold_idx
        fold_records.append(fold_metrics)

    oof_pred = np.array([LABELS[idx] for idx in oof_proba.argmax(axis=1)])
    oof_metrics = evaluate(f"{name}_oof", y_train, oof_pred, proba=oof_proba)

    model = build_model(params, RANDOM_STATE)
    fit_kwargs = {"cat_features": categorical} if name.startswith("catboost") else {"categorical_feature": categorical}
    model.fit(X_train, y_train, **fit_kwargs)
    test_proba = align_proba(model.predict_proba(X_test), model.classes_)
    test_pred = np.array([LABELS[idx] for idx in test_proba.argmax(axis=1)])
    test_metrics = evaluate(f"{name}_test", y_test, test_pred, proba=test_proba)

    details = classification_details(y_test, test_pred, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / f"cm_{name}_test.png",
        f"Confusion Matrix: {name} Test",
    )
    return fold_records, [oof_metrics, test_metrics]


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    train_df = add_semester_ratio_features(train_df)
    test_df = add_semester_ratio_features(test_df)

    catboost_params = load_json(METRICS_DIR / "catboost_tuned_best_params.json")
    lightgbm_params = load_json(METRICS_DIR / "lightgbm_tuned_best_params.json")

    all_folds = []
    all_metrics = []
    for name, params, prepare_xy, build_model in [
        ("catboost_tuned_semester", catboost_params, prepare_catboost_xy, make_catboost),
        ("lightgbm_tuned_semester", lightgbm_params, prepare_lightgbm_xy, make_lightgbm),
    ]:
        fold_records, metrics = run_member(name, train_df, test_df, params, prepare_xy, build_model)
        all_folds.extend(fold_records)
        all_metrics.extend(metrics)

    fold_path = METRICS_DIR / "tree_semester_features_5fold.csv"
    save_metrics_table(all_folds, fold_path)

    metrics_path = METRICS_DIR / "tree_semester_features_metrics.csv"
    save_metrics_table(all_metrics, metrics_path)

    save_metrics_table_figure(
        pd.DataFrame(all_metrics)[
            [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
            ]
        ],
        FIGURE_DIR / "tree_semester_features_summary.png",
        "Tuned Tree Models + Semester Features",
    )

    print(pd.DataFrame(all_metrics).to_string(index=False))
    print(f"saved fold metrics: {fold_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
