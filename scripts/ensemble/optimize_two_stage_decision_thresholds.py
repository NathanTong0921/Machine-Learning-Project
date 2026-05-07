"""Tune decision thresholds for the cumulative two-stage decomposition.

Declaration:
- This script appends a calibrated two-stage experiment.
- It does not overwrite the original one-stage multiclass scripts.
- Thresholds are selected only on out-of-fold training predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import search_cumulative_link_models as cumulative
from src.config import FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns
from src.visualization import save_metrics_table_figure


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


def conditional_from_recomposed(proba):
    p_not_dropout = np.clip(1.0 - proba[:, 0], 0.0, 1.0)
    p_graduate_given_not_dropout = np.divide(
        proba[:, 2],
        np.maximum(p_not_dropout, 1e-12),
        out=np.zeros_like(p_not_dropout),
        where=p_not_dropout > 1e-12,
    )
    return p_not_dropout, np.clip(p_graduate_given_not_dropout, 0.0, 1.0)


def threshold_predictions(p_not_dropout, p_graduate_given_not_dropout, threshold_not_dropout, threshold_graduate):
    pred = np.full(len(p_not_dropout), "Dropout", dtype=object)
    not_dropout_mask = p_not_dropout >= threshold_not_dropout
    pred[not_dropout_mask] = np.where(
        p_graduate_given_not_dropout[not_dropout_mask] >= threshold_graduate,
        "Graduate",
        "Enrolled",
    )
    return pred


def evaluate_predictions(name, y_true, y_pred, y_prob):
    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, labels=LABELS)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["model"] = name
    for label in LABELS:
        key = label.lower()
        metrics[f"{key}_precision"] = report[label]["precision"]
        metrics[f"{key}_recall"] = report[label]["recall"]
        metrics[f"{key}_f1"] = report[label]["f1-score"]
    return metrics


def evaluate_argmax(name, y_true, proba):
    pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
    return evaluate_predictions(name, y_true, pred, proba)


def load_member_specs():
    catboost_params = cumulative.load_json(METRICS_DIR / "catboost_tuned_best_params.json")
    lightgbm_params = cumulative.load_json(METRICS_DIR / "lightgbm_tuned_best_params.json")
    hist = cumulative.load_json(METRICS_DIR / "hist_column_sampling_best_params.json")["models"]["lightgbm"]["params"]
    classic = cumulative.load_json(METRICS_DIR / "classic_boosting_best_params.json")["models"]["gbdt_log_loss"]["best_params"]
    lightgbm_hist_params = {**lightgbm_params, "max_bin": hist["max_bin"], "colsample_bytree": hist["column_fraction"]}
    return [
        ("catboost_cumulative", "catboost", catboost_params),
        ("lightgbm_hist_cumulative", "lightgbm", lightgbm_hist_params),
        ("gbdt_cumulative", "gbdt", classic),
    ]


def get_cumulative_member_predictions(train_df, test_df, cv, n_jobs, quick):
    numeric_features, categorical_features = split_feature_columns(train_df.drop(columns=[TARGET_COL]).columns)
    members = load_member_specs()
    if quick:
        members = [member for member in members if member[0] == "lightgbm_hist_cumulative"]

    names = []
    oof_probas = []
    test_probas = []
    for name, kind, params in members:
        if kind == "catboost":
            oof, test, _, _ = cumulative.fit_predict_catboost(params, train_df, test_df, cv)
        elif kind == "lightgbm":
            oof, test, _, _ = cumulative.fit_predict_lightgbm(name, params, train_df, test_df, cv, n_jobs)
        else:
            oof, test, _, _ = cumulative.fit_predict_gbdt(
                params,
                train_df,
                test_df,
                cv,
                numeric_features,
                categorical_features,
            )
        names.append(name)
        oof_probas.append(oof)
        test_probas.append(test)
    return names, oof_probas, test_probas


def search_ensemble_weights(member_names, oof_probas, y_true, step):
    if len(member_names) == 1:
        return np.array([1.0]), 0.0, pd.DataFrame([{"oof_f1_macro": f1_score(y_true, np.array([LABELS[idx] for idx in oof_probas[0].argmax(axis=1)]), average="macro", zero_division=0), f"weight_{member_names[0]}": 1.0}])
    return cumulative.search_weights(member_names, oof_probas, y_true, step)


def threshold_grid(coarse_step):
    values = np.round(np.arange(0.20, 0.801, coarse_step), 4)
    for threshold_not_dropout in values:
        for threshold_graduate in values:
            yield float(threshold_not_dropout), float(threshold_graduate)


def refine_grid(best_not_dropout, best_graduate):
    not_dropout_values = np.round(
        np.arange(max(0.05, best_not_dropout - 0.05), min(0.95, best_not_dropout + 0.0501), 0.005),
        4,
    )
    graduate_values = np.round(
        np.arange(max(0.05, best_graduate - 0.05), min(0.95, best_graduate + 0.0501), 0.005),
        4,
    )
    for threshold_not_dropout in not_dropout_values:
        for threshold_graduate in graduate_values:
            yield float(threshold_not_dropout), float(threshold_graduate)


def tune_thresholds(y_true, p_not_dropout, p_graduate_given_not_dropout, coarse_step):
    records = []
    best = {"score": -np.inf, "threshold_not_dropout": None, "threshold_graduate": None}
    for threshold_not_dropout, threshold_graduate in threshold_grid(coarse_step):
        pred = threshold_predictions(p_not_dropout, p_graduate_given_not_dropout, threshold_not_dropout, threshold_graduate)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        records.append(
            {
                "stage_a_threshold_not_dropout": threshold_not_dropout,
                "stage_b_threshold_graduate": threshold_graduate,
                "oof_f1_macro": float(score),
            }
        )
        if score > best["score"]:
            best = {
                "score": float(score),
                "threshold_not_dropout": threshold_not_dropout,
                "threshold_graduate": threshold_graduate,
            }

    for threshold_not_dropout, threshold_graduate in refine_grid(best["threshold_not_dropout"], best["threshold_graduate"]):
        pred = threshold_predictions(p_not_dropout, p_graduate_given_not_dropout, threshold_not_dropout, threshold_graduate)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        records.append(
            {
                "stage_a_threshold_not_dropout": threshold_not_dropout,
                "stage_b_threshold_graduate": threshold_graduate,
                "oof_f1_macro": float(score),
            }
        )
        if score > best["score"]:
            best = {
                "score": float(score),
                "threshold_not_dropout": threshold_not_dropout,
                "threshold_graduate": threshold_graduate,
            }
    return best, pd.DataFrame(records).sort_values("oof_f1_macro", ascending=False)


def load_one_stage_reference():
    references = []
    paths = [
        (METRICS_DIR / "machine_only_ensemble_metrics.csv", "machine_only_ensemble_test"),
        (METRICS_DIR / "catboost_tuned_metrics.csv", "catboost_tuned"),
        (METRICS_DIR / "hist_column_sampling_metrics.csv", "lightgbm"),
        (METRICS_DIR / "oof_probability_ensemble_semester_members_metrics.csv", "oof_probability_ensemble_semester_members_test"),
    ]
    for path, model in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if model in set(df["model"]):
            row = df[df["model"] == model].iloc[0].to_dict()
            row["reference_source"] = str(path)
            references.append(row)
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    member_names, oof_probas, test_probas = get_cumulative_member_predictions(train_df, test_df, cv, args.n_jobs, args.quick)
    weights, weight_score, weight_grid = search_ensemble_weights(member_names, oof_probas, y_train, args.weight_step)
    ensemble_oof = np.tensordot(weights, np.stack(oof_probas, axis=0), axes=(0, 0))
    ensemble_test = np.tensordot(weights, np.stack(test_probas, axis=0), axes=(0, 0))

    p_not_dropout_oof, p_graduate_oof = conditional_from_recomposed(ensemble_oof)
    p_not_dropout_test, p_graduate_test = conditional_from_recomposed(ensemble_test)
    threshold_best, threshold_grid_df = tune_thresholds(y_train, p_not_dropout_oof, p_graduate_oof, args.threshold_step)

    tuned_oof_pred = threshold_predictions(
        p_not_dropout_oof,
        p_graduate_oof,
        threshold_best["threshold_not_dropout"],
        threshold_best["threshold_graduate"],
    )
    tuned_test_pred = threshold_predictions(
        p_not_dropout_test,
        p_graduate_test,
        threshold_best["threshold_not_dropout"],
        threshold_best["threshold_graduate"],
    )

    metric_records = [
        evaluate_argmax("two_stage_uncalibrated_oof", y_train, ensemble_oof),
        evaluate_argmax("two_stage_uncalibrated_test", y_test, ensemble_test),
        evaluate_predictions("two_stage_threshold_tuned_oof", y_train, tuned_oof_pred, ensemble_oof),
        evaluate_predictions("two_stage_threshold_tuned_test", y_test, tuned_test_pred, ensemble_test),
    ]

    reference_records = load_one_stage_reference()
    compare_records = []
    for row in reference_records:
        compare_records.append(
            {
                "model": row["model"],
                "source": "one_stage_reference",
                "accuracy": row.get("accuracy"),
                "f1_macro": row.get("f1_macro"),
                "f1_weighted": row.get("f1_weighted"),
                "fmi": row.get("fmi"),
                "roc_auc_ovr_macro": row.get("roc_auc_ovr_macro"),
                "dropout_accuracy": row.get("dropout_accuracy", row.get("dropout_recall")),
                "enrolled_accuracy": row.get("enrolled_accuracy", row.get("enrolled_recall")),
                "graduate_accuracy": row.get("graduate_accuracy", row.get("graduate_recall")),
            }
        )
    for row in metric_records:
        compare_records.append(
            {
                "model": row["model"],
                "source": "two_stage_threshold_experiment",
                "accuracy": row.get("accuracy"),
                "f1_macro": row.get("f1_macro"),
                "f1_weighted": row.get("f1_weighted"),
                "fmi": row.get("fmi"),
                "roc_auc_ovr_macro": row.get("roc_auc_ovr_macro"),
                "dropout_accuracy": row.get("dropout_accuracy"),
                "enrolled_accuracy": row.get("enrolled_accuracy"),
                "graduate_accuracy": row.get("graduate_accuracy"),
            }
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_table(metric_records, METRICS_DIR / "two_stage_threshold_tuned_metrics.csv")
    save_metrics_table(compare_records, METRICS_DIR / "two_stage_vs_one_stage_comparison.csv")
    save_metrics_table(weight_grid.to_dict(orient="records"), METRICS_DIR / "two_stage_threshold_tuned_weight_grid.csv")
    save_metrics_table(threshold_grid_df.to_dict(orient="records"), METRICS_DIR / "two_stage_threshold_tuned_threshold_grid.csv")

    payload = {
        "members": member_names,
        "weights": {name: float(weight) for name, weight in zip(member_names, weights)},
        "weight_oof_f1_macro": float(weight_score),
        "thresholds": {
            "stage_a_threshold_not_dropout": float(threshold_best["threshold_not_dropout"]),
            "stage_b_threshold_graduate": float(threshold_best["threshold_graduate"]),
            "oof_f1_macro": float(threshold_best["score"]),
        },
        "cv_splits": args.cv,
        "selection": "Thresholds selected on OOF macro-F1; test evaluated once.",
        "interpretation": {
            "lower_stage_a_threshold": "More conservative about predicting Dropout; lets more samples reach Stage B.",
            "higher_stage_b_threshold": "More conservative about predicting Graduate; gives Enrolled more room.",
        },
    }
    (METRICS_DIR / "two_stage_threshold_tuned_best.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    compare_df = pd.DataFrame(compare_records).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    save_metrics_table_figure(
        compare_df[
            [
                "model",
                "source",
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
                "dropout_accuracy",
                "enrolled_accuracy",
                "graduate_accuracy",
            ]
        ].head(12),
        FIGURE_DIR / "two_stage_vs_one_stage_comparison.png",
        "Two-Stage Threshold Tuning vs One-Stage Models",
    )

    print(pd.DataFrame(metric_records).sort_values("f1_macro", ascending=False).to_string(index=False))
    print("\n=== Comparison ===")
    print(compare_df.to_string(index=False))
    print(f"\nthresholds: {payload['thresholds']}")
    print(f"weights: {payload['weights']}")


if __name__ == "__main__":
    main()
