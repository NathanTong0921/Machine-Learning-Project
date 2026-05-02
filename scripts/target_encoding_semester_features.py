"""Evaluate target encoding plus semester-ratio features.

Declaration:
- This script tests whether better categorical encoding and domain-shaped semester
  ratio features can improve macro-F1.
- It uses 5-fold Stratified CV on the processed training split and also reports
  a final train/test evaluation for comparison with existing result tables.
- TargetEncoder is placed inside the pipeline, so each CV fold fits encodings on
  the training fold only.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_metrics_table_figure


RATIO_FEATURES = [
    "first_approved_rate",
    "second_approved_rate",
    "first_eval_pass_rate",
    "second_eval_pass_rate",
    "grade_delta",
    "approved_delta",
    "tuition_scholarship_interaction",
    "debtor_tuition_interaction",
]


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


def build_preprocessor(feature_columns):
    categorical_features = [col for col in feature_columns if col in CATEGORICAL_COLUMNS]
    numeric_features = [col for col in feature_columns if col not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
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
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def build_models(feature_columns):
    return {
        "targetenc_semester_logistic": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_columns)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "targetenc_semester_linear_svm": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_columns)),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        max_iter=10000,
                        dual="auto",
                    ),
                ),
            ]
        ),
        "targetenc_semester_histgb": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(feature_columns)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        l2_regularization=0.01,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(y_true, y_pred, y_prob=None):
    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, labels=LABELS if y_prob is not None else None)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    for label in LABELS:
        metrics[f"{label.lower()}_precision"] = report[label]["precision"]
        metrics[f"{label.lower()}_recall"] = report[label]["recall"]
        metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]
    return metrics


def predict_proba_if_available(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def run_cv(train_df):
    train_df = add_semester_ratio_features(train_df)
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    feature_columns = X.columns.tolist()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    records = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)

        for model_name, model in build_models(feature_columns).items():
            print(f"[PROGRESS] fold={fold_idx}/5 model={model_name}", flush=True)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            y_prob = predict_proba_if_available(model, X_valid)
            metrics = evaluate_predictions(y_valid, y_pred, y_prob=y_prob)
            metrics["fold"] = fold_idx
            metrics["model"] = model_name
            records.append(metrics)

    return pd.DataFrame(records)


def summarize_cv(fold_df):
    metric_cols = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "roc_auc_ovr_macro",
        "enrolled_recall",
        "enrolled_f1",
    ]
    available_metric_cols = [col for col in metric_cols if col in fold_df.columns]
    summary_df = fold_df.groupby("model", as_index=False)[available_metric_cols].agg(["mean", "std"])
    summary_df.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in summary_df.columns.to_flat_index()
    ]
    return summary_df


def run_test_eval(train_df, test_df):
    train_df = add_semester_ratio_features(train_df)
    test_df = add_semester_ratio_features(test_df)
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    feature_columns = X_train.columns.tolist()

    records = []
    for model_name, model in build_models(feature_columns).items():
        print(f"[PROGRESS] final_test model={model_name}", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = predict_proba_if_available(model, X_test)
        metrics = evaluate_predictions(y_test, y_pred, y_prob=y_prob)
        metrics["model"] = model_name
        records.append(metrics)

        details = classification_details(y_test, y_pred, labels=LABELS)
        save_confusion_matrix(
            details["confusion_matrix"],
            LABELS,
            f"reports/figures/cm_{model_name}.png",
            f"Confusion Matrix: {model_name}",
        )
    return pd.DataFrame(records)


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    fold_df = run_cv(train_df)
    cv_path = METRICS_DIR / "target_encoding_semester_5fold.csv"
    save_metrics_table(fold_df.to_dict(orient="records"), cv_path)

    summary_df = summarize_cv(fold_df)
    summary_path = METRICS_DIR / "target_encoding_semester_5fold_summary.csv"
    save_metrics_table(summary_df.to_dict(orient="records"), summary_path)

    test_df_metrics = run_test_eval(train_df, test_df)
    test_path_out = METRICS_DIR / "target_encoding_semester_test_metrics.csv"
    save_metrics_table(test_df_metrics.to_dict(orient="records"), test_path_out)

    save_metrics_table_figure(
        test_df_metrics[
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
        "reports/figures/target_encoding_semester_test_summary.png",
        "Target Encoding + Semester Features",
    )

    print("\n[5-fold summary]")
    print(summary_df.to_string(index=False))
    print("\n[test metrics]")
    print(test_df_metrics.to_string(index=False))
    print(f"saved cv folds: {cv_path}")
    print(f"saved cv summary: {summary_path}")
    print(f"saved test metrics: {test_path_out}")


if __name__ == "__main__":
    main()
