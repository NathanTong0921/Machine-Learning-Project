"""Run a 10-fold hyperparameter search for the CatBoost baseline.

Declaration:
- This script performs a focused 10-fold search for CatBoost on the student outcome dataset.
- It uses CatBoost's native categorical-feature handling instead of one-hot encoding.

Tips:
- Read macro-F1 first.
- This search is deliberately moderate in size so it is still practical to run.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_metrics_table_figure


PARAM_GRID = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05],
    "iterations": [300, 500],
    "l2_leaf_reg": [3, 6, 9],
    "random_strength": [0.5, 1.0],
}


def save_status(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def progress_print(message):
    print(f"[PROGRESS] {message}", flush=True)


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL]).copy()
    y_test = test_df[TARGET_COL]

    cat_cols = [col for col in X_train.columns if col in CATEGORICAL_COLUMNS]
    cat_idx = [X_train.columns.get_loc(col) for col in cat_cols]

    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    param_list = list(ParameterGrid(PARAM_GRID))
    total_folds = len(param_list) * cv.get_n_splits()
    status_path = METRICS_DIR / "catboost_tuned_status.json"
    cv_results_path = METRICS_DIR / "catboost_tuned_cv_results.csv"

    cv_records = []
    best_params = None
    best_cv_score = -np.inf
    completed_folds = 0

    save_status(
        status_path,
        {
            "stage": "cv_search",
            "total_param_combinations": len(param_list),
            "total_folds": total_folds,
            "completed_folds": completed_folds,
        },
    )
    progress_print(f"start param_combinations={len(param_list)} total_folds={total_folds}")

    outer_iter = tqdm(param_list, desc="CatBoost param grid", unit="combo") if tqdm is not None else param_list
    for combo_idx, params in enumerate(outer_iter, start=1):
        fold_scores = []
        fold_iter = cv.split(X_train, y_train)
        for fold_idx, (train_idx, valid_idx) in enumerate(fold_iter, start=1):
            X_tr = X_train.iloc[train_idx].copy()
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[valid_idx].copy()
            y_va = y_train.iloc[valid_idx]

            model = CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="TotalF1:average=Macro",
                auto_class_weights="Balanced",
                verbose=0,
                random_seed=42,
                thread_count=-1,
                **params,
            )
            model.fit(X_tr, y_tr, cat_features=cat_idx)
            pred = model.predict(X_va).ravel()
            score = f1_score(y_va, pred, average="macro", zero_division=0)
            fold_scores.append(score)
            completed_folds += 1

            save_status(
                status_path,
                {
                    "stage": "cv_search",
                    "current_params": params,
                    "current_combo_index": combo_idx,
                    "current_fold": fold_idx,
                    "total_param_combinations": len(param_list),
                    "total_folds": total_folds,
                    "completed_folds": completed_folds,
                    "progress_ratio": completed_folds / total_folds,
                    "best_cv_score_so_far": best_cv_score,
                    "best_params_so_far": best_params,
                },
            )
            progress_print(
                f"combo={combo_idx}/{len(param_list)} fold={fold_idx}/10 "
                f"score={score:.4f} completed={completed_folds}/{total_folds}"
            )

        mean_score = float(np.mean(fold_scores))
        record = {
            **params,
            "cv_mean_f1_macro": mean_score,
            "cv_std_f1_macro": float(np.std(fold_scores)),
        }
        cv_records.append(record)
        save_metrics_table(cv_records, cv_results_path)

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_params = params
            progress_print(f"new_best combo={combo_idx} cv_mean_f1={mean_score:.4f} params={params}")

    best_model = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=42,
        thread_count=-1,
        **best_params,
    )
    save_status(
        status_path,
        {
            "stage": "refit",
            "best_params": best_params,
            "best_cv_score_f1_macro": best_cv_score,
        },
    )
    best_model.fit(X_train, y_train, cat_features=cat_idx)
    y_pred = best_model.predict(X_test).ravel()
    y_prob = best_model.predict_proba(X_test)

    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = "catboost_tuned"
    metrics["cv_best_score_f1_macro"] = best_cv_score
    save_metrics_table([metrics], METRICS_DIR / "catboost_tuned_metrics.csv")

    details = classification_details(y_test, y_pred, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / "cm_catboost_tuned.png",
        "Confusion Matrix: CatBoost Tuned",
    )

    with open(METRICS_DIR / "catboost_tuned_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    save_metrics_table_figure(
        pd.DataFrame([metrics])[
            [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
                "cv_best_score_f1_macro",
            ]
        ],
        FIGURE_DIR / "catboost_tuned_summary.png",
        "CatBoost 10-Fold Search Summary",
    )

    save_status(
        status_path,
        {
            "stage": "completed",
            "best_params": best_params,
            "best_cv_score_f1_macro": best_cv_score,
            "paths": {
                "metrics": str(METRICS_DIR / "catboost_tuned_metrics.csv"),
                "best_params": str(METRICS_DIR / "catboost_tuned_best_params.json"),
                "cv_results": str(cv_results_path),
            },
        },
    )

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"best params: {best_params}")
    print(f"saved metrics: {METRICS_DIR / 'catboost_tuned_metrics.csv'}")
    print(f"saved params: {METRICS_DIR / 'catboost_tuned_best_params.json'}")


if __name__ == "__main__":
    main()
