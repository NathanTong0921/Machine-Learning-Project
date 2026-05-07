"""Compare cross-seed stacking against cross-seed soft averaging.

Declaration:
- This script uses only classical machine-learning members by default.
- It excludes MLP, Transformer, TabPFN, and other deep-learning members.
- Base predictions are generated out-of-fold; the meta learner sees only OOF
  probabilities on the training set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import tune_machine_only_ensemble as base
from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns
from src.visualization import save_metrics_table_figure


DEFAULT_MEMBERS = ["catboost_tuned", "lightgbm_hist_column_best", "gbdt_log_loss"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 62, 82])
    parser.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    return parser.parse_args()


def load_member_specs():
    classic = base.load_json(METRICS_DIR / "classic_boosting_best_params.json")["models"]
    hist = base.load_json(METRICS_DIR / "hist_column_sampling_best_params.json")["models"]
    catboost_params = base.load_json(METRICS_DIR / "catboost_tuned_best_params.json")
    lightgbm_params = base.load_json(METRICS_DIR / "lightgbm_tuned_best_params.json")
    lightgbm_hist_params = {
        **lightgbm_params,
        "max_bin": hist["lightgbm"]["params"]["max_bin"],
        "colsample_bytree": hist["lightgbm"]["params"]["column_fraction"],
    }
    return {
        "catboost_tuned": ("catboost", catboost_params),
        "lightgbm_hist_column_best": ("lightgbm", lightgbm_hist_params),
        "gbdt_log_loss": ("sklearn", classic["gbdt_log_loss"]["best_params"]),
        "hist_gradient_boosting": ("sklearn", classic["hist_gradient_boosting"]["best_params"]),
        "adaboost_stumps": ("sklearn", classic["adaboost_stumps"]["best_params"]),
    }


def evaluate(name, y_true, proba):
    pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
    metrics = classification_metrics(y_true, pred, y_prob=proba, labels=LABELS)
    report = classification_report(y_true, pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["model"] = name
    for label in LABELS:
        key = label.lower()
        metrics[f"{key}_precision"] = report[label]["precision"]
        metrics[f"{key}_recall"] = report[label]["recall"]
        metrics[f"{key}_f1"] = report[label]["f1-score"]
    return metrics


def sklearn_oof_test(name, params, train_df, test_df, cv, seed, numeric_features, categorical_features):
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    use_sample_weight = name in {"adaboost_stumps", "gbdt_log_loss"}
    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] seed={seed} member={name} fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        model = base.make_sklearn_member(name, params, seed + fold_idx, numeric_features, categorical_features)
        fit_kwargs = {"model__sample_weight": sample_weight[train_idx]} if use_sample_weight else {}
        model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
        oof[valid_idx] = base.align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)
    model = base.make_sklearn_member(name, params, seed, numeric_features, categorical_features)
    fit_kwargs = {"model__sample_weight": sample_weight} if use_sample_weight else {}
    model.fit(X, y, **fit_kwargs)
    return oof, base.align_proba(model.predict_proba(X_test), model.classes_)


def catboost_oof_test(params, train_df, test_df, cv, seed):
    X, y, cat_idx = base.prepare_catboost_xy(train_df)
    X_test, _, _ = base.prepare_catboost_xy(test_df)
    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] seed={seed} member=catboost_tuned fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        model = base.make_catboost(params, seed + fold_idx)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_idx)
        oof[valid_idx] = base.align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)
    model = base.make_catboost(params, seed)
    model.fit(X, y, cat_features=cat_idx)
    return oof, base.align_proba(model.predict_proba(X_test), model.classes_)


def lightgbm_oof_test(name, params, train_df, test_df, cv, seed, n_jobs):
    X, y, cat_cols = base.prepare_category_xy(train_df)
    X_test, _, _ = base.prepare_category_xy(test_df)
    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] seed={seed} member={name} fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        model = base.make_lightgbm(params, seed + fold_idx, n_jobs)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], categorical_feature=cat_cols)
        oof[valid_idx] = base.align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)
    model = base.make_lightgbm(params, seed, n_jobs)
    model.fit(X, y, categorical_feature=cat_cols)
    return oof, base.align_proba(model.predict_proba(X_test), model.classes_)


def main():
    args = parse_args()
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    numeric_features, categorical_features = split_feature_columns(train_df.drop(columns=[TARGET_COL]).columns)
    specs = load_member_specs()

    records = []
    feature_blocks_oof = []
    feature_blocks_test = []
    all_oof_probas = []
    all_test_probas = []
    seed_oof_probas = {}
    seed_test_probas = {}

    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=seed)
        seed_oof_blocks = []
        seed_test_blocks = []
        for member in args.members:
            kind, params = specs[member]
            if kind == "catboost":
                oof, test = catboost_oof_test(params, train_df, test_df, cv, seed)
            elif kind == "lightgbm":
                oof, test = lightgbm_oof_test(member, params, train_df, test_df, cv, seed, args.n_jobs)
            else:
                oof, test = sklearn_oof_test(member, params, train_df, test_df, cv, seed, numeric_features, categorical_features)
            records.append(evaluate(f"{member}_seed{seed}_test", y_test, test))
            feature_blocks_oof.append(oof)
            feature_blocks_test.append(test)
            all_oof_probas.append(oof)
            all_test_probas.append(test)
            seed_oof_blocks.append(oof)
            seed_test_blocks.append(test)
        seed_oof_probas[seed] = np.mean(np.stack(seed_oof_blocks, axis=0), axis=0)
        seed_test_probas[seed] = np.mean(np.stack(seed_test_blocks, axis=0), axis=0)
        records.append(evaluate(f"soft_average_seed{seed}_oof", y_train, seed_oof_probas[seed]))
        records.append(evaluate(f"soft_average_seed{seed}_test", y_test, seed_test_probas[seed]))

    cross_seed_oof = np.mean(np.stack(all_oof_probas, axis=0), axis=0)
    cross_seed_test = np.mean(np.stack(all_test_probas, axis=0), axis=0)
    records.append(evaluate("cross_seed_soft_average_oof", y_train, cross_seed_oof))
    records.append(evaluate("cross_seed_soft_average_test", y_test, cross_seed_test))

    stack_X_train = np.concatenate(feature_blocks_oof, axis=1)
    stack_X_test = np.concatenate(feature_blocks_test, axis=1)
    meta = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)
    meta_cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    meta_oof = cross_val_predict(meta, stack_X_train, y_train, cv=meta_cv, method="predict_proba", n_jobs=args.n_jobs)
    meta_oof = base.align_proba(meta_oof, meta.classes_ if hasattr(meta, "classes_") else LABELS)
    records.append(evaluate("cross_seed_stacking_oof", y_train, meta_oof))
    meta.fit(stack_X_train, y_train)
    meta_test = base.align_proba(meta.predict_proba(stack_X_test), meta.classes_)
    records.append(evaluate("cross_seed_stacking_test", y_test, meta_test))

    metrics_df = pd.DataFrame(records).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    save_metrics_table(metrics_df.to_dict(orient="records"), METRICS_DIR / "cross_seed_stacking_metrics.csv")

    payload = {
        "members": args.members,
        "seeds": args.seeds,
        "cv_splits": args.cv,
        "excluded": ["MLP", "Transformer", "TabPFN", "all deep-learning members"],
        "feature_count": int(stack_X_train.shape[1]),
        "meta_model": "LogisticRegression(max_iter=5000, class_weight='balanced')",
        "metric_notes": {
            "fmi": "Fowlkes-Mallows index from true and predicted class labels.",
            "class_accuracy": "Per-class accuracy is TP / true class count, equivalent to class recall.",
        },
    }
    (METRICS_DIR / "cross_seed_stacking_config.json").write_text(
        json.dumps(base.json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    keep = metrics_df[
        metrics_df["model"].isin(
            [
                "cross_seed_soft_average_test",
                "cross_seed_stacking_test",
                "cross_seed_soft_average_oof",
                "cross_seed_stacking_oof",
            ]
        )
    ]
    save_metrics_table_figure(
        keep[
            [
                "model",
                "accuracy",
                "fmi",
                "f1_macro",
                "f1_weighted",
                "roc_auc_ovr_macro",
                "dropout_accuracy",
                "enrolled_accuracy",
                "graduate_accuracy",
            ]
        ],
        base.FIGURE_DIR / "cross_seed_stacking_summary.png",
        "Cross-Seed Stacking",
    )

    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
