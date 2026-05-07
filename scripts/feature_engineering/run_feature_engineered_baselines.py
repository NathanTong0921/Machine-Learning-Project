"""Train feature-engineered baseline models and save metrics.

Declaration:
- This script extends the baseline family with explicit course-style feature engineering.
- It is intended to test whether simple models plus better features can close part of
  the gap to stronger baselines or advanced models.

Tips:
- Run `scripts/preprocessing/prepare_data.py` first.
- Compare these results with plain logistic regression and random forest.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns
from src.visualization import save_confusion_matrix


def build_poly_preprocessor(numeric_features, categorical_features, degree=2):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_dense_preprocessor(numeric_features, categorical_features):
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
        ]
    )


def build_feature_engineered_models(numeric_features, categorical_features):
    return {
        "logistic_poly2": Pipeline(
            steps=[
                ("preprocessor", build_poly_preprocessor(numeric_features, categorical_features, degree=2)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "logistic_rbf_nystroem": Pipeline(
            steps=[
                ("preprocessor", build_dense_preprocessor(numeric_features, categorical_features)),
                ("rbf_features", Nystroem(kernel="rbf", gamma=0.2, n_components=400, random_state=RANDOM_STATE)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    numeric_features, categorical_features = split_feature_columns(X_train.columns)

    records = []
    for model_name, pipeline in build_feature_engineered_models(numeric_features, categorical_features).items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

        metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
        metrics["model"] = model_name
        records.append(metrics)

        details = classification_details(y_test, y_pred, labels=LABELS)
        save_confusion_matrix(
            details["confusion_matrix"],
            LABELS,
            f"reports/figures/cm_{model_name}.png",
            f"Confusion Matrix: {model_name}",
        )
        print(f"\n[{model_name}]")
        print(details["report"])

    save_metrics_table(records, METRICS_DIR / "feature_engineered_baseline_metrics.csv")
    print(f"saved metrics: {METRICS_DIR / 'feature_engineered_baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
