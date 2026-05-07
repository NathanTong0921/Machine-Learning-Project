"""Run a 10-fold hyperparameter search for the LightGBM baseline.

Declaration:
- This script performs a focused 10-fold search for LightGBM on the student outcome dataset.
- It uses LightGBM's categorical-feature support with progress/status outputs.

Tips:
- Read macro-F1 first.
- This search mirrors the CatBoost tuning workflow so results are easy to compare.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.visualization import save_confusion_matrix, save_metrics_table_figure


PARAM_GRID = {
    "num_leaves": [31, 63],
    "max_depth": [-1, 8, 12],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [300, 500],
    "min_child_samples": [20, 40],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

LIGHTGBM_N_JOBS = int(os.environ.get("LIGHTGBM_N_JOBS", "108"))
PARAM_COLUMNS = [
    "colsample_bytree",
    "learning_rate",
    "max_depth",
    "min_child_samples",
    "n_estimators",
    "num_leaves",
    "subsample",
]
PARAM_CASTERS = {
    "colsample_bytree": float,
    "learning_rate": float,
    "max_depth": int,
    "min_child_samples": int,
    "n_estimators": int,
    "num_leaves": int,
    "subsample": float,
}


def save_status(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def progress_print(message):
    print(f"[PROGRESS] {message}", flush=True)


def load_existing_cv_results(path):
    if not path.exists() or path.stat().st_size == 0:
        return [], None, -np.inf

    results_df = pd.read_csv(path)
    if results_df.empty:
        return [], None, -np.inf

    records = results_df.to_dict("records")
    best_idx = results_df["cv_mean_f1_macro"].astype(float).idxmax()
    best_row = results_df.loc[best_idx]
    best_params = {col: PARAM_CASTERS[col](best_row[col]) for col in PARAM_COLUMNS}
    best_score = float(best_row["cv_mean_f1_macro"])
    return records, best_params, best_score


def main():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL]).copy()
    y_test = test_df[TARGET_COL]

    cat_cols = [col for col in X_train.columns if col in CATEGORICAL_COLUMNS]
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    param_list = list(ParameterGrid(PARAM_GRID))
    total_folds = len(param_list) * cv.get_n_splits()
    status_path = METRICS_DIR / "lightgbm_tuned_status.json"
    cv_results_path = METRICS_DIR / "lightgbm_tuned_cv_results.csv"

    cv_records, best_params, best_cv_score = load_existing_cv_results(cv_results_path)
    completed_combos = min(len(cv_records), len(param_list))
    completed_folds = completed_combos * cv.get_n_splits()
    remaining_params = param_list[completed_combos:]

    save_status(
        status_path,
        {
            "stage": "cv_search",
            "total_param_combinations": len(param_list),
            "total_folds": total_folds,
            "completed_folds": completed_folds,
            "completed_param_combinations": completed_combos,
            "resume_from_combo_index": completed_combos + 1 if remaining_params else None,
            "best_cv_score_so_far": best_cv_score,
            "best_params_so_far": best_params,
            "lightgbm_n_jobs": LIGHTGBM_N_JOBS,
        },
    )
    progress_print(
        f"start param_combinations={len(param_list)} total_folds={total_folds} "
        f"completed_combos={completed_combos} lightgbm_n_jobs={LIGHTGBM_N_JOBS}"
    )

    outer_iter = tqdm(
        remaining_params,
        desc="LightGBM param grid",
        unit="combo",
        initial=completed_combos,
        total=len(param_list),
    ) if tqdm is not None else remaining_params
    for combo_idx, params in enumerate(outer_iter, start=completed_combos + 1):
        fold_scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
            X_tr = X_train.iloc[train_idx].copy()
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[valid_idx].copy()
            y_va = y_train.iloc[valid_idx]

            model = LGBMClassifier(
                objective="multiclass",
                class_weight="balanced",
                random_state=42,
                verbosity=-1,
                n_jobs=LIGHTGBM_N_JOBS,
                **params,
            )
            model.fit(X_tr, y_tr, categorical_feature=cat_cols)
            pred = model.predict(X_va)
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
                    "lightgbm_n_jobs": LIGHTGBM_N_JOBS,
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

    best_model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
        n_jobs=LIGHTGBM_N_JOBS,
        **best_params,
    )
    save_status(
        status_path,
        {
            "stage": "refit",
            "best_params": best_params,
            "best_cv_score_f1_macro": best_cv_score,
            "lightgbm_n_jobs": LIGHTGBM_N_JOBS,
        },
    )
    best_model.fit(X_train, y_train, categorical_feature=cat_cols)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = "lightgbm_tuned"
    metrics["cv_best_score_f1_macro"] = best_cv_score
    save_metrics_table([metrics], METRICS_DIR / "lightgbm_tuned_metrics.csv")

    details = classification_details(y_test, y_pred, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / "cm_lightgbm_tuned.png",
        "Confusion Matrix: LightGBM Tuned",
    )

    with open(METRICS_DIR / "lightgbm_tuned_best_params.json", "w", encoding="utf-8") as f:
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
        FIGURE_DIR / "lightgbm_tuned_summary.png",
        "LightGBM 10-Fold Search Summary",
    )

    save_status(
        status_path,
        {
            "stage": "completed",
            "best_params": best_params,
            "best_cv_score_f1_macro": best_cv_score,
            "lightgbm_n_jobs": LIGHTGBM_N_JOBS,
            "paths": {
                "metrics": str(METRICS_DIR / "lightgbm_tuned_metrics.csv"),
                "best_params": str(METRICS_DIR / "lightgbm_tuned_best_params.json"),
                "cv_results": str(cv_results_path),
            },
        },
    )

    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"best params: {best_params}")
    print(f"saved metrics: {METRICS_DIR / 'lightgbm_tuned_metrics.csv'}")
    print(f"saved params: {METRICS_DIR / 'lightgbm_tuned_best_params.json'}")


if __name__ == "__main__":
    main()
