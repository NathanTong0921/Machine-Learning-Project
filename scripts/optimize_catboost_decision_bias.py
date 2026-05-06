"""Optimize class-specific decision bias for tuned CatBoost.

Declaration:
- This script does not tune on the final test set.
- It first builds 5-fold out-of-fold probabilities on the processed training set.
- It then searches class-specific log-probability biases on OOF predictions to maximize macro-F1.
- Finally, it refits CatBoost on the full training set and evaluates raw vs biased decisions on the held-out test set.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_metrics_table_figure


BIAS_VALUES = np.round(np.arange(-0.60, 0.65, 0.05), 2)
EPS = 1e-12


def load_best_params():
    path = METRICS_DIR / "catboost_tuned_best_params.json"
    if not path.exists():
        raise FileNotFoundError("Missing catboost_tuned_best_params.json. Run tune_catboost_baseline.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_xy(df):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    cat_cols = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    cat_idx = [X.columns.get_loc(col) for col in cat_cols]
    for col in cat_cols:
        X[col] = X[col].astype(str)
    return X, y, cat_idx


def build_model(params, seed=RANDOM_STATE):
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=seed,
        thread_count=-1,
        **params,
    )


def align_proba(proba, classes):
    aligned = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    class_to_col = {str(label): idx for idx, label in enumerate(classes)}
    for out_idx, label in enumerate(LABELS):
        aligned[:, out_idx] = proba[:, class_to_col[label]]
    return aligned


def apply_bias(proba, bias):
    logits = np.log(np.clip(proba, EPS, 1.0)) + bias.reshape(1, -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    biased_proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    pred_idx = biased_proba.argmax(axis=1)
    pred_labels = np.array([LABELS[idx] for idx in pred_idx])
    return pred_labels, biased_proba


def evaluate(name, y_true, y_pred, y_prob=None):
    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, labels=LABELS if y_prob is not None else None)
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["model"] = name
    for label in LABELS:
        metrics[f"{label.lower()}_precision"] = report[label]["precision"]
        metrics[f"{label.lower()}_recall"] = report[label]["recall"]
        metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]
    return metrics


def make_oof_predictions(X, y, cat_idx, params):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(X), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] oof fold={fold_idx}/5", flush=True)
        model = build_model(params, seed=RANDOM_STATE + fold_idx)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_idx)
        proba = model.predict_proba(X.iloc[valid_idx])
        oof_proba[valid_idx] = align_proba(proba, model.classes_)
    return oof_proba


def search_bias(y_true, oof_proba):
    records = []
    best_score = -np.inf
    best_bias = None
    best_pred = None

    # Dropout bias is fixed at zero because adding the same constant to all classes is invariant.
    for enrolled_bias in BIAS_VALUES:
        for graduate_bias in BIAS_VALUES:
            bias = np.array([0.0, enrolled_bias, graduate_bias], dtype=float)
            pred, _ = apply_bias(oof_proba, bias)
            score = f1_score(y_true, pred, average="macro", zero_division=0)
            records.append(
                {
                    "dropout_bias": 0.0,
                    "enrolled_bias": enrolled_bias,
                    "graduate_bias": graduate_bias,
                    "oof_f1_macro": score,
                }
            )
            if score > best_score:
                best_score = score
                best_bias = bias
                best_pred = pred

    return best_bias, best_score, best_pred, pd.DataFrame(records).sort_values("oof_f1_macro", ascending=False)


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    params = load_best_params()

    X_train, y_train, cat_idx = prepare_xy(train_df)
    X_test, y_test, _ = prepare_xy(test_df)

    oof_proba = make_oof_predictions(X_train, y_train, cat_idx, params)
    raw_oof_pred = np.array([LABELS[idx] for idx in oof_proba.argmax(axis=1)])
    raw_oof_metrics = evaluate("catboost_tuned_raw_oof", y_train, raw_oof_pred, y_prob=oof_proba)

    best_bias, best_score, biased_oof_pred, grid_df = search_bias(y_train, oof_proba)
    biased_oof_pred, biased_oof_proba = apply_bias(oof_proba, best_bias)
    biased_oof_metrics = evaluate("catboost_tuned_biased_oof", y_train, biased_oof_pred, y_prob=biased_oof_proba)

    print(f"[PROGRESS] best_oof_bias={best_bias.tolist()} best_oof_f1_macro={best_score:.6f}", flush=True)
    model = build_model(params)
    model.fit(X_train, y_train, cat_features=cat_idx)
    test_proba = align_proba(model.predict_proba(X_test), model.classes_)
    raw_test_pred = np.array([LABELS[idx] for idx in test_proba.argmax(axis=1)])
    biased_test_pred, biased_test_proba = apply_bias(test_proba, best_bias)

    raw_test_metrics = evaluate("catboost_tuned_raw_test", y_test, raw_test_pred, y_prob=test_proba)
    biased_test_metrics = evaluate("catboost_tuned_biased_test", y_test, biased_test_pred, y_prob=biased_test_proba)

    for model_name, pred in [
        ("catboost_tuned_raw_test", raw_test_pred),
        ("catboost_tuned_biased_test", biased_test_pred),
    ]:
        details = classification_details(y_test, pred, labels=LABELS)
        save_confusion_matrix(
            details["confusion_matrix"],
            LABELS,
            FIGURE_DIR / f"cm_{model_name}.png",
            f"Confusion Matrix: {model_name}",
        )

    metrics = [raw_oof_metrics, biased_oof_metrics, raw_test_metrics, biased_test_metrics]
    metrics_path = METRICS_DIR / "catboost_decision_bias_metrics.csv"
    save_metrics_table(metrics, metrics_path)

    grid_path = METRICS_DIR / "catboost_decision_bias_grid.csv"
    save_metrics_table(grid_df.to_dict(orient="records"), grid_path)

    bias_payload = {
        "labels": LABELS,
        "bias": {label: float(value) for label, value in zip(LABELS, best_bias)},
        "oof_f1_macro": float(best_score),
        "params": params,
    }
    bias_path = METRICS_DIR / "catboost_decision_bias_best.json"
    with open(bias_path, "w", encoding="utf-8") as f:
        json.dump(bias_payload, f, ensure_ascii=False, indent=2)

    save_metrics_table_figure(
        pd.DataFrame([raw_test_metrics, biased_test_metrics])[
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
        FIGURE_DIR / "catboost_decision_bias_summary.png",
        "CatBoost Decision Bias",
    )

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved bias grid: {grid_path}")
    print(f"saved best bias: {bias_path}")


if __name__ == "__main__":
    main()
