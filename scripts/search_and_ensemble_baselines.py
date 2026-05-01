"""Run grid search for all baseline models, then build voting and stacking ensembles.

Declaration:
- This script performs a broad but still reportable brute-force hyperparameter search
  over all classical baselines, then trains ensemble models on the tuned estimators.
- It is the course-aligned "model selection + ensemble learning" bridge.

Tips:
- This script can take a while because it uses cross-validation and multiple model families.
- Read macro-F1 first when comparing tuned models and ensembles.
"""

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import get_baseline_search_spaces
from src.config import FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_details, classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor
from src.visualization import save_confusion_matrix, save_confusion_matrix_grid, save_metrics_table_figure


MEDIUM_PARAM_GRIDS = {
    "logistic_regression": {"model__C": [0.2, 1.0, 5.0]},
    "decision_tree": {
        "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [1, 3],
        "model__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [8, None],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
        "model__class_weight": ["balanced"],
    },
    "gradient_boosting": {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
        "model__subsample": [0.8, 1.0],
    },
    "adaboost": {
        "model__estimator__max_depth": [1, 2],
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.5, 1.0],
    },
    "extra_trees": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [8, None],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt"],
        "model__class_weight": ["balanced"],
    },
    "svm_rbf": {
        "model__C": [1.0, 2.0, 5.0],
        "model__gamma": ["scale", 0.1],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search and ensemble baselines")
    parser.add_argument("--mode", choices=["medium", "full"], default="medium")
    parser.add_argument("--save-per-model", action="store_true")
    parser.add_argument("--flag-path", type=str, default="")
    parser.add_argument("--loocv-top-k", type=int, default=0)
    return parser.parse_args()


def flag_print(message):
    print(f"[FLAG] {message}", flush=True)


def save_status(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_search_spaces(mode, random_state):
    search_spaces = get_baseline_search_spaces(random_state=random_state)
    if mode == "medium":
        for model_name in search_spaces:
            search_spaces[model_name]["param_grid"] = MEDIUM_PARAM_GRIDS[model_name]
    return search_spaces


def evaluate_model(model_name, estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
    metrics = classification_metrics(y_test, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = model_name
    details = classification_details(y_test, y_pred, labels=LABELS)
    save_confusion_matrix(
        details["confusion_matrix"],
        LABELS,
        FIGURE_DIR / f"cm_{model_name}.png",
        f"Confusion Matrix: {model_name}",
    )
    return metrics


def kfold_weights(best_estimators, X_train, y_train, n_splits, random_state):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    records = []
    weights = []

    for model_name, estimator in best_estimators.items():
        y_pred = cross_val_predict(
            estimator,
            X_train,
            y_train,
            cv=cv,
            n_jobs=-1,
            method="predict",
        )
        try:
            y_prob = cross_val_predict(
                estimator,
                X_train,
                y_train,
                cv=cv,
                n_jobs=-1,
                method="predict_proba",
            )
        except Exception:
            y_prob = None

        metrics = classification_metrics(y_train, y_pred, y_prob=y_prob, labels=LABELS)
        raw_weight = max(metrics["f1_macro"], 1e-6)
        weights.append(raw_weight)
        records.append(
            {
                "model": model_name,
                f"{n_splits}fold_accuracy": metrics["accuracy"],
                f"{n_splits}fold_precision_macro": metrics["precision_macro"],
                f"{n_splits}fold_recall_macro": metrics["recall_macro"],
                f"{n_splits}fold_f1_macro": metrics["f1_macro"],
                f"{n_splits}fold_f1_weighted": metrics["f1_weighted"],
                f"{n_splits}fold_roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
                f"{n_splits}fold_weight_raw": raw_weight,
            }
        )

    weight_sum = float(sum(weights))
    normalized = [w / weight_sum for w in weights]
    for record, weight in zip(records, normalized):
        record[f"{n_splits}fold_weight_normalized"] = weight

    weight_map = {record["model"]: record[f"{n_splits}fold_weight_normalized"] for record in records}
    return records, weight_map


def main():
    args = parse_args()
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    grid_n_splits = 10
    weight_n_splits = 10
    cv = StratifiedKFold(n_splits=grid_n_splits, shuffle=True, random_state=RANDOM_STATE)
    search_spaces = get_search_spaces(mode=args.mode, random_state=RANDOM_STATE)

    prefix = f"{args.mode}_baseline"
    metrics_path = METRICS_DIR / f"{prefix}_gridsearch_metrics.csv"
    best_params_path = METRICS_DIR / f"{prefix}_gridsearch_best_params.csv"
    kfold_weight_path = METRICS_DIR / f"{prefix}_{weight_n_splits}fold_weights.csv"
    ensemble_path = METRICS_DIR / f"{prefix}_ensemble_metrics.csv"
    grid_fig_path = FIGURE_DIR / f"{prefix}_gridsearch_summary.png"
    loocv_fig_path = FIGURE_DIR / f"{prefix}_loocv_weights.png"
    ensemble_fig_path = FIGURE_DIR / f"{prefix}_ensemble_summary.png"
    ensemble_cm_grid_path = FIGURE_DIR / f"{prefix}_ensemble_confusion_grid.png"
    status_path = Path(args.flag_path) if args.flag_path else METRICS_DIR / f"{prefix}_status.json"

    tuned_records = []
    search_records = []
    best_estimators = {}
    completed_models = []
    pending_models = list(search_spaces.keys())

    save_status(
        status_path,
        {
            "mode": args.mode,
            "stage": "grid_search",
            "grid_n_splits": grid_n_splits,
            "weight_n_splits": weight_n_splits,
            "completed_models": completed_models,
            "pending_models": pending_models,
        },
    )
    flag_print(f"started mode={args.mode} models={len(search_spaces)} status={status_path}")

    for model_name, spec in search_spaces.items():
        flag_print(f"grid_search_start model={model_name}")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", spec["model"]),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec["param_grid"],
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_estimators[model_name] = search.best_estimator_

        test_metrics = evaluate_model(model_name, search.best_estimator_, X_test, y_test)
        test_metrics["cv_best_score_f1_macro"] = search.best_score_
        tuned_records.append(test_metrics)

        search_records.append(
            {
                "model": model_name,
                "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                "cv_best_score_f1_macro": search.best_score_,
            }
        )
        completed_models.append(model_name)
        pending_models = [name for name in search_spaces.keys() if name not in completed_models]
        best_so_far = max(tuned_records, key=lambda x: x["f1_macro"])
        save_status(
            status_path,
            {
                "mode": args.mode,
                "stage": "grid_search",
                "grid_n_splits": grid_n_splits,
                "weight_n_splits": weight_n_splits,
                "completed_models": completed_models,
                "pending_models": pending_models,
                "last_completed_model": model_name,
                "best_so_far": best_so_far,
            },
        )
        if args.save_per_model:
            save_metrics_table(tuned_records, metrics_path)
            save_metrics_table(search_records, best_params_path)
        flag_print(
            f"grid_search_done model={model_name} {grid_n_splits}fold_cv_f1={search.best_score_:.4f} "
            f"test_f1={test_metrics['f1_macro']:.4f}"
        )

    ranked_names = [row["model"] for row in sorted(search_records, key=lambda x: x["cv_best_score_f1_macro"], reverse=True)]
    default_top_k = 4 if args.mode == "medium" else len(ranked_names)
    top_k = args.loocv_top_k or default_top_k
    selected_names = ranked_names[:top_k]
    selected_estimators = {name: best_estimators[name] for name in selected_names}

    save_status(
        status_path,
        {
            "mode": args.mode,
            "stage": f"{weight_n_splits}fold_weighting",
            "grid_n_splits": grid_n_splits,
            "weight_n_splits": weight_n_splits,
            "completed_models": completed_models,
            "pending_models": [],
            "selected_for_weighting": selected_names,
        },
    )
    flag_print(f"{weight_n_splits}fold_weighting_start top_k={top_k} selected={selected_names}")

    kfold_records, kfold_weight_map = kfold_weights(
        selected_estimators,
        X_train,
        y_train,
        n_splits=weight_n_splits,
        random_state=RANDOM_STATE,
    )
    ordered_names = list(selected_estimators.keys())
    ordered_estimators = [(name, best_estimators[name]) for name in ordered_names]
    ordered_weights = [kfold_weight_map[name] for name in ordered_names]
    flag_print(f"{weight_n_splits}fold_weighting_done weights={kfold_weight_map}")

    soft_voting = VotingClassifier(
        estimators=ordered_estimators,
        voting="soft",
        weights=ordered_weights,
        n_jobs=-1,
    )
    hard_voting = VotingClassifier(
        estimators=ordered_estimators,
        voting="hard",
        weights=ordered_weights,
        n_jobs=-1,
    )
    stacking = StackingClassifier(
        estimators=ordered_estimators,
        final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    ensemble_models = {
        "voting_soft": soft_voting,
        "voting_hard": hard_voting,
        "stacking": stacking,
    }
    ensemble_records = []
    save_status(
        status_path,
        {
            "mode": args.mode,
            "stage": "ensemble_training",
            "grid_n_splits": grid_n_splits,
            "weight_n_splits": weight_n_splits,
            "selected_for_weighting": selected_names,
            "weight_map": kfold_weight_map,
        },
    )
    for model_name, estimator in ensemble_models.items():
        flag_print(f"ensemble_start model={model_name}")
        estimator.fit(X_train, y_train)
        metrics = evaluate_model(model_name, estimator, X_test, y_test)
        ensemble_records.append(metrics)
        if args.save_per_model:
            save_metrics_table(ensemble_records, ensemble_path)
        flag_print(f"ensemble_done model={model_name} test_f1={metrics['f1_macro']:.4f}")

    tuned_df = pd.DataFrame(tuned_records).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    search_df = pd.DataFrame(search_records).sort_values("cv_best_score_f1_macro", ascending=False).reset_index(drop=True)
    ensemble_df = pd.DataFrame(ensemble_records).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    save_metrics_table(tuned_df.to_dict(orient="records"), metrics_path)
    save_metrics_table(search_df.to_dict(orient="records"), best_params_path)
    save_metrics_table(kfold_records, kfold_weight_path)
    save_metrics_table(ensemble_df.to_dict(orient="records"), ensemble_path)

    save_metrics_table_figure(
        tuned_df[
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
        grid_fig_path,
        "Baseline Grid Search Summary",
    )
    save_metrics_table_figure(
        pd.DataFrame(kfold_records)[
            [
                "model",
                f"{weight_n_splits}fold_accuracy",
                f"{weight_n_splits}fold_f1_macro",
                f"{weight_n_splits}fold_roc_auc_ovr_macro",
                f"{weight_n_splits}fold_weight_normalized",
            ]
        ],
        loocv_fig_path,
        f"Baseline {weight_n_splits}-Fold Voting Weights",
    )
    save_metrics_table_figure(
        ensemble_df[
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
        ensemble_fig_path,
        "Baseline Ensemble Summary",
    )

    image_paths = [
        FIGURE_DIR / "cm_voting_soft.png",
        FIGURE_DIR / "cm_voting_hard.png",
        FIGURE_DIR / "cm_stacking.png",
    ]
    save_confusion_matrix_grid(
        image_paths=image_paths,
        titles=["Soft Voting", "Hard Voting", "Stacking"],
        output_path=ensemble_cm_grid_path,
        nrows=1,
        ncols=3,
    )

    final_best_single = tuned_df.iloc[0].to_dict()
    final_best_ensemble = ensemble_df.iloc[0].to_dict()
    save_status(
        status_path,
        {
            "mode": args.mode,
            "stage": "completed",
            "grid_n_splits": grid_n_splits,
            "weight_n_splits": weight_n_splits,
            "selected_for_weighting": selected_names,
            "weight_map": kfold_weight_map,
            "best_single_model": final_best_single,
            "best_ensemble_model": final_best_ensemble,
            "paths": {
                "grid_metrics": str(metrics_path),
                "best_params": str(best_params_path),
                "weight_metrics": str(kfold_weight_path),
                "ensemble_metrics": str(ensemble_path),
            },
        },
    )
    flag_print(f"completed best_single={final_best_single['model']} best_ensemble={final_best_ensemble['model']}")

    print("\n=== Tuned Baselines ===")
    print(tuned_df.to_string(index=False))
    print("\n=== Ensembles ===")
    print(ensemble_df.to_string(index=False))
    print(f"saved tuned baseline metrics: {metrics_path}")
    print(f"saved best params: {best_params_path}")
    print(f"saved {weight_n_splits}-fold weights: {kfold_weight_path}")
    print(f"saved ensemble metrics: {ensemble_path}")
    print(f"saved status: {status_path}")


if __name__ == "__main__":
    main()
