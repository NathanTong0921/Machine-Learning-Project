"""Measure Application order contribution in Logistic Regression encodings.

Outputs two complementary views:
- coefficient share: absolute/L2 coefficient mass of Application-order derived
  features among all Logistic Regression coefficients after preprocessing.
- grouped permutation importance: test metric drop after shuffling the original
  Application-order derived columns together.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics
from src.preprocessing import build_preprocessor, split_feature_columns


APP_ORDER_COL = "Application order"
OUTPUT_DIR = METRICS_DIR / "baseline"
OUT_SUMMARY = OUTPUT_DIR / "logistic_application_order_contribution.csv"
OUT_REPEAT = OUTPUT_DIR / "logistic_application_order_group_permutation_repeats.csv"
OUT_BEST = OUTPUT_DIR / "logistic_application_order_contribution_summary.json"

C_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
N_REPEATS = 30


def add_thermometer_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_out = train_df.copy()
    test_out = test_df.copy()
    columns = []
    for threshold in range(1, 10):
        col = f"Application order >= {threshold}"
        train_out[col] = (train_out[APP_ORDER_COL] >= threshold).astype(float)
        test_out[col] = (test_out[APP_ORDER_COL] >= threshold).astype(float)
        columns.append(col)
    train_out = train_out.drop(columns=[APP_ORDER_COL])
    test_out = test_out.drop(columns=[APP_ORDER_COL])
    return train_out, test_out, columns


def smoothed_class_rates(frame: pd.DataFrame, alpha: float):
    counts = frame[TARGET_COL].value_counts().reindex(LABELS, fill_value=0).astype(float)
    total = float(len(frame))
    return (counts + alpha) / (total + alpha * len(LABELS))


def add_oof_target_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits=5, alpha=10.0):
    train_out = train_df.copy()
    test_out = test_df.copy()
    columns = [f"Application order te {label.lower()}" for label in LABELS]
    for col in columns:
        train_out[col] = np.nan

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fit_idx, valid_idx in cv.split(train_df, train_df[TARGET_COL]):
        fold_train = train_df.iloc[fit_idx]
        fold_global = smoothed_class_rates(fold_train, alpha)
        maps = {}
        for order_value, group in fold_train.groupby(APP_ORDER_COL):
            rates = smoothed_class_rates(group, alpha)
            weight = len(group) / (len(group) + alpha)
            maps[order_value] = weight * rates + (1.0 - weight) * fold_global

        for row_idx in valid_idx:
            order_value = train_df.iloc[row_idx][APP_ORDER_COL]
            rates = maps.get(order_value, fold_global)
            for label, col in zip(LABELS, columns):
                train_out.iat[row_idx, train_out.columns.get_loc(col)] = rates[label]

    global_rates = smoothed_class_rates(train_df, alpha)
    full_maps = {}
    for order_value, group in train_df.groupby(APP_ORDER_COL):
        rates = smoothed_class_rates(group, alpha)
        weight = len(group) / (len(group) + alpha)
        full_maps[order_value] = weight * rates + (1.0 - weight) * global_rates

    for col in columns:
        test_out[col] = np.nan
    for row_idx, order_value in enumerate(test_df[APP_ORDER_COL]):
        rates = full_maps.get(order_value, global_rates)
        for label, col in zip(LABELS, columns):
            test_out.iat[row_idx, test_out.columns.get_loc(col)] = rates[label]

    train_out[columns] = train_out[columns].astype(float)
    test_out[columns] = test_out[columns].astype(float)
    train_out = train_out.drop(columns=[APP_ORDER_COL])
    test_out = test_out.drop(columns=[APP_ORDER_COL])
    return train_out, test_out, columns


def make_pipeline(numeric_features, categorical_features):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            (
                "model",
                LogisticRegression(
                    max_iter=6000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_best_model(train_df, numeric_features, categorical_features):
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        make_pipeline(numeric_features, categorical_features),
        param_grid={"model__C": C_GRID},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_["model__C"], grid.best_score_


def feature_names(model):
    return model.named_steps["preprocessor"].get_feature_names_out()


def coefficient_share(model, group_tokens):
    names = feature_names(model)
    coefs = model.named_steps["model"].coef_
    group_mask = np.array([any(token in name for token in group_tokens) for name in names])
    abs_total = np.abs(coefs).sum()
    l2_total = np.sqrt((coefs**2).sum())
    group_abs = np.abs(coefs[:, group_mask]).sum()
    group_l2 = np.sqrt((coefs[:, group_mask] ** 2).sum())
    return {
        "transformed_feature_count": int(len(names)),
        "group_feature_count": int(group_mask.sum()),
        "coef_abs_share": float(group_abs / abs_total) if abs_total else 0.0,
        "coef_l2_share": float(group_l2 / l2_total) if l2_total else 0.0,
        "top_group_coefficients": json.dumps(top_group_coefficients(names, coefs, group_mask), ensure_ascii=False),
    }


def top_group_coefficients(names, coefs, group_mask, top_k=8):
    rows = []
    classes = list(model_classes())
    for class_idx, class_name in enumerate(classes):
        for name, coef, keep in zip(names, coefs[class_idx], group_mask):
            if keep:
                rows.append(
                    {
                        "class": class_name,
                        "feature": str(name),
                        "coef": float(coef),
                        "abs_coef": float(abs(coef)),
                    }
                )
    rows = sorted(rows, key=lambda row: row["abs_coef"], reverse=True)
    return rows[:top_k]


def model_classes():
    return LABELS


def evaluate(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)
    return classification_metrics(y, pred, y_prob=prob, labels=LABELS)


def grouped_permutation(model, X_test, y_test, group_columns, rng_seed):
    rng = np.random.default_rng(rng_seed)
    base = evaluate(model, X_test, y_test)
    rows = []
    for repeat in range(N_REPEATS):
        shuffled = X_test.copy()
        order = rng.permutation(len(shuffled))
        for col in group_columns:
            shuffled[col] = shuffled[col].to_numpy()[order]
        permuted = evaluate(model, shuffled, y_test)
        rows.append(
            {
                "repeat": repeat + 1,
                "accuracy_drop": base["accuracy"] - permuted["accuracy"],
                "f1_macro_drop": base["f1_macro"] - permuted["f1_macro"],
                "auc_drop": base["roc_auc_ovr_macro"] - permuted["roc_auc_ovr_macro"],
                "enrolled_f1_drop": base["enrolled_f1"] - permuted["enrolled_f1"],
            }
        )
    return base, rows


def build_variants(train_df, test_df):
    variants = []
    base_numeric, base_categorical = split_feature_columns(train_df.drop(columns=[TARGET_COL]).columns)
    variants.append(
        {
            "model": "logistic_app_order_scalar",
            "train": train_df,
            "test": test_df,
            "numeric": base_numeric,
            "categorical": base_categorical,
            "group_columns": [APP_ORDER_COL],
            "group_tokens": [APP_ORDER_COL],
        }
    )

    onehot_numeric = [col for col in base_numeric if col != APP_ORDER_COL]
    onehot_categorical = list(base_categorical) + [APP_ORDER_COL]
    variants.append(
        {
            "model": "logistic_app_order_onehot",
            "train": train_df,
            "test": test_df,
            "numeric": onehot_numeric,
            "categorical": onehot_categorical,
            "group_columns": [APP_ORDER_COL],
            "group_tokens": [APP_ORDER_COL],
        }
    )

    thermo_train, thermo_test, thermo_cols = add_thermometer_features(train_df, test_df)
    thermo_numeric, thermo_categorical = split_feature_columns(thermo_train.drop(columns=[TARGET_COL]).columns)
    variants.append(
        {
            "model": "logistic_app_order_thermometer",
            "train": thermo_train,
            "test": thermo_test,
            "numeric": thermo_numeric,
            "categorical": thermo_categorical,
            "group_columns": thermo_cols,
            "group_tokens": thermo_cols,
        }
    )

    te_train, te_test, te_cols = add_oof_target_encoding(train_df, test_df)
    te_numeric, te_categorical = split_feature_columns(te_train.drop(columns=[TARGET_COL]).columns)
    variants.append(
        {
            "model": "logistic_app_order_oof_target_encoding",
            "train": te_train,
            "test": te_test,
            "numeric": te_numeric,
            "categorical": te_categorical,
            "group_columns": te_cols,
            "group_tokens": te_cols,
        }
    )
    return variants


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    repeat_rows = []
    for variant in build_variants(train_df, test_df):
        model, best_c, cv_score = fit_best_model(variant["train"], variant["numeric"], variant["categorical"])
        X_test = variant["test"].drop(columns=[TARGET_COL])
        y_test = variant["test"][TARGET_COL]
        base_metrics, perm_rows = grouped_permutation(
            model,
            X_test,
            y_test,
            variant["group_columns"],
            rng_seed=RANDOM_STATE + len(summary_rows),
        )
        coef_stats = coefficient_share(model, variant["group_tokens"])
        repeat_df = pd.DataFrame(perm_rows)
        row = {
            "model": variant["model"],
            "best_C": best_c,
            "cv_f1_macro_mean": cv_score,
            "test_accuracy": base_metrics["accuracy"],
            "test_f1_macro": base_metrics["f1_macro"],
            "test_auc": base_metrics["roc_auc_ovr_macro"],
            "test_enrolled_f1": base_metrics["enrolled_f1"],
            **coef_stats,
            "perm_accuracy_drop_mean": repeat_df["accuracy_drop"].mean(),
            "perm_accuracy_drop_std": repeat_df["accuracy_drop"].std(ddof=1),
            "perm_f1_macro_drop_mean": repeat_df["f1_macro_drop"].mean(),
            "perm_f1_macro_drop_std": repeat_df["f1_macro_drop"].std(ddof=1),
            "perm_auc_drop_mean": repeat_df["auc_drop"].mean(),
            "perm_auc_drop_std": repeat_df["auc_drop"].std(ddof=1),
            "perm_enrolled_f1_drop_mean": repeat_df["enrolled_f1_drop"].mean(),
            "perm_enrolled_f1_drop_std": repeat_df["enrolled_f1_drop"].std(ddof=1),
            "group_columns": json.dumps(variant["group_columns"], ensure_ascii=False),
        }
        summary_rows.append(row)
        for perm_row in perm_rows:
            repeat_rows.append({"model": variant["model"], **perm_row})
        print(
            f"{variant['model']}: coef_abs_share={row['coef_abs_share']:.4f}, "
            f"perm_f1_drop={row['perm_f1_macro_drop_mean']:.4f}, "
            f"perm_auc_drop={row['perm_auc_drop_mean']:.4f}"
        )

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    pd.DataFrame(repeat_rows).to_csv(OUT_REPEAT, index=False)
    with open(OUT_BEST, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"saved summary: {OUT_SUMMARY}")
    print(f"saved repeats: {OUT_REPEAT}")
    print(f"saved json: {OUT_BEST}")


if __name__ == "__main__":
    main()
