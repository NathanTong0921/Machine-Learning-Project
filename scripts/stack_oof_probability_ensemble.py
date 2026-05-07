"""Train a regularized meta-learner on OOF probability features.

Declaration:
- Base-model predictions come from oof_probability_ensemble.py.
- Hyperparameters are selected only with CV on OOF-derived training features.
- The held-out test set is evaluated once with the selected meta-learner.
- This is an exploratory stacking layer; the raw OOF ensemble remains the
  conservative reference unless the held-out macro-F1 improves.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURE_DIR, LABELS, METRICS_DIR, RANDOM_STATE
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_metrics_table_figure


EPS = 1e-12
CV_SPLITS = 5
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
CLASS_WEIGHTS = [None, "balanced"]
FEATURE_SETS = [
    "member_proba",
    "member_proba_log",
    "member_proba_margin_entropy",
    "member_proba_log_margin_entropy",
    "member_plus_weighted_ensemble",
    "full",
]


def load_probability_cache():
    path = METRICS_DIR / "oof_probability_ensemble_probas.npz"
    if not path.exists():
        raise FileNotFoundError(
            "Missing oof_probability_ensemble_probas.npz. "
            "Run `python scripts/oof_probability_ensemble.py` first."
        )
    return path, np.load(path, allow_pickle=True)


def flatten_member_proba(probas):
    # probas shape: members x samples x classes
    return probas.transpose(1, 0, 2).reshape(probas.shape[1], -1)


def entropy_features(probas):
    clipped = np.clip(probas, EPS, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=2).T


def margin_features(probas):
    sorted_proba = np.sort(probas, axis=2)
    return (sorted_proba[:, :, -1] - sorted_proba[:, :, -2]).T


def single_proba_extras(proba):
    clipped = np.clip(proba, EPS, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1, keepdims=True)
    sorted_proba = np.sort(proba, axis=1)
    margin = (sorted_proba[:, -1] - sorted_proba[:, -2]).reshape(-1, 1)
    return np.hstack([proba, np.log(clipped), entropy, margin])


def make_features(member_probas, weighted_ensemble_proba, feature_set):
    blocks = [flatten_member_proba(member_probas)]

    if feature_set in {"member_proba_log", "member_proba_log_margin_entropy", "full"}:
        blocks.append(flatten_member_proba(np.log(np.clip(member_probas, EPS, 1.0))))

    if feature_set in {"member_proba_margin_entropy", "member_proba_log_margin_entropy", "full"}:
        blocks.append(margin_features(member_probas))
        blocks.append(entropy_features(member_probas))

    if feature_set in {"member_plus_weighted_ensemble", "full"}:
        blocks.append(single_proba_extras(weighted_ensemble_proba))

    return np.hstack(blocks)


def align_proba(proba, classes):
    aligned = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    class_to_col = {str(label): idx for idx, label in enumerate(classes)}
    for out_idx, label in enumerate(LABELS):
        aligned[:, out_idx] = proba[:, class_to_col[label]]
    return aligned


def build_model(c_value, class_weight):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
    )


def predict_aligned_proba(model, X):
    proba = model.predict_proba(X)
    classes = model.named_steps["logisticregression"].classes_
    return align_proba(proba, classes)


def evaluate(name, y_true, pred, proba=None):
    metrics = classification_metrics(y_true, pred, y_prob=proba, labels=LABELS if proba is not None else None)
    report = classification_report(y_true, pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["model"] = name
    for label in LABELS:
        metrics[f"{label.lower()}_precision"] = report[label]["precision"]
        metrics[f"{label.lower()}_recall"] = report[label]["recall"]
        metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]
    return metrics


def evaluate_raw(name, y_true, proba):
    pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
    return evaluate(name, y_true, pred, proba=proba)


def cross_validate_configs(oof_member_probas, ensemble_oof, y_train):
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    records = []

    for feature_set in FEATURE_SETS:
        X_meta = make_features(oof_member_probas, ensemble_oof, feature_set)
        for c_value in C_VALUES:
            for class_weight in CLASS_WEIGHTS:
                fold_scores = []
                for train_idx, valid_idx in cv.split(X_meta, y_train):
                    model = build_model(c_value, class_weight)
                    model.fit(X_meta[train_idx], y_train[train_idx])
                    pred = model.predict(X_meta[valid_idx])
                    fold_scores.append(f1_score(y_train[valid_idx], pred, average="macro", zero_division=0))

                records.append(
                    {
                        "feature_set": feature_set,
                        "c_value": c_value,
                        "class_weight": "none" if class_weight is None else class_weight,
                        "cv_mean_f1_macro": float(np.mean(fold_scores)),
                        "cv_std_f1_macro": float(np.std(fold_scores)),
                        "n_features": int(X_meta.shape[1]),
                    }
                )

    grid_df = pd.DataFrame(records).sort_values("cv_mean_f1_macro", ascending=False)
    return grid_df


def main():
    cache_path, data = load_probability_cache()
    member_names = data["member_names"].astype(str).tolist()
    y_train = data["y_train"].astype(str)
    y_test = data["y_test"].astype(str)
    oof_member_probas = data["oof_probas"]
    test_member_probas = data["test_probas"]
    ensemble_oof = data["ensemble_oof"]
    ensemble_test = data["ensemble_test"]

    grid_df = cross_validate_configs(oof_member_probas, ensemble_oof, y_train)
    best = grid_df.iloc[0].to_dict()
    best_class_weight = None if best["class_weight"] == "none" else best["class_weight"]

    X_train_meta = make_features(oof_member_probas, ensemble_oof, best["feature_set"])
    X_test_meta = make_features(test_member_probas, ensemble_test, best["feature_set"])
    model = build_model(float(best["c_value"]), best_class_weight)
    model.fit(X_train_meta, y_train)

    train_proba = predict_aligned_proba(model, X_train_meta)
    test_proba = predict_aligned_proba(model, X_test_meta)
    train_pred = np.array([LABELS[idx] for idx in train_proba.argmax(axis=1)])
    test_pred = np.array([LABELS[idx] for idx in test_proba.argmax(axis=1)])

    metrics = [
        evaluate_raw("oof_probability_ensemble_raw_oof", y_train, ensemble_oof),
        evaluate("oof_probability_stacker_insample_oof", y_train, train_pred, proba=train_proba),
        evaluate_raw("oof_probability_ensemble_raw_test", y_test, ensemble_test),
        evaluate("oof_probability_stacker_test", y_test, test_pred, proba=test_proba),
    ]
    metrics[1]["meta_cv_f1_macro"] = float(best["cv_mean_f1_macro"])
    metrics[1]["meta_cv_std_f1_macro"] = float(best["cv_std_f1_macro"])
    metrics[3]["meta_cv_f1_macro"] = float(best["cv_mean_f1_macro"])
    metrics[3]["meta_cv_std_f1_macro"] = float(best["cv_std_f1_macro"])

    for model_name, pred in [
        ("oof_probability_ensemble_raw_test", np.array([LABELS[idx] for idx in ensemble_test.argmax(axis=1)])),
        ("oof_probability_stacker_test", test_pred),
    ]:
        details = classification_details(y_test, pred, labels=LABELS)
        save_confusion_matrix(
            details["confusion_matrix"],
            LABELS,
            FIGURE_DIR / f"cm_{model_name}.png",
            f"Confusion Matrix: {model_name}",
        )

    metrics_path = METRICS_DIR / "oof_probability_stacking_metrics.csv"
    save_metrics_table(metrics, metrics_path)

    grid_path = METRICS_DIR / "oof_probability_stacking_grid.csv"
    save_metrics_table(grid_df.to_dict(orient="records"), grid_path)

    best_payload = {
        "member_names": member_names,
        "source_probability_cache": str(cache_path),
        "best_config": best,
        "selection_rule": "5-fold CV on OOF-derived meta-features; held-out test used only once for evaluation",
        "note": "Insample OOF stacker metrics are diagnostic only; use meta_cv_f1_macro and held-out test metrics for claims.",
    }
    best_path = METRICS_DIR / "oof_probability_stacking_best.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    save_metrics_table_figure(
        pd.DataFrame([metrics[2], metrics[3]])[
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
        FIGURE_DIR / "oof_probability_stacking_summary.png",
        "OOF Probability Stacking",
    )

    print(f"[PROGRESS] best_meta_config={best_payload['best_config']}", flush=True)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved grid: {grid_path}")
    print(f"saved best config: {best_path}")


if __name__ == "__main__":
    main()
