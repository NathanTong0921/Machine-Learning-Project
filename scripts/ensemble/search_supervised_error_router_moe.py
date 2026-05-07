"""Search a supervised error-predicting router for a pure-ML Tabular MoE.

This is MoE v2 after the unsupervised GMM-router side experiment.

Router target:
    For each training row, use out-of-fold expert probabilities and assign the
    pseudo-label to the expert with the lowest true-class negative log loss.

Router models:
    Classical sklearn classifiers trained on Appendix-aligned dense tabular
    preprocessing. No DL / TabPFN member is used.

Aggregation:
    Hard routing selects the top-1 predicted expert.
    Soft routing weights all expert probabilities by router probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tune_machine_only_ensemble as machine_only

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns


EPS = 1e-12


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    return json_safe(metrics)


def make_router(name, numeric_features, categorical_features, seed, n_jobs):
    if name == "logistic_router":
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        )
    elif name == "extra_trees_router":
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )
    elif name == "random_forest_router":
        model = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(f"Unknown router: {name}")
    return Pipeline(
        steps=[
            ("preprocessor", machine_only.dense_preprocessor(numeric_features, categorical_features)),
            ("router", model),
        ]
    )


def align_router_proba(proba, classes, n_members):
    aligned = np.zeros((proba.shape[0], n_members), dtype=float)
    for col_idx, member_idx in enumerate(classes):
        aligned[:, int(member_idx)] = proba[:, col_idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sums, out=np.full_like(aligned, 1.0 / n_members), where=row_sums > 0)


def true_class_losses(oof_probas, y_true):
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    y_idx = np.array([label_to_idx[str(label)] for label in y_true], dtype=np.int64)
    stacked = np.stack(oof_probas, axis=0)
    true_probs = stacked[:, np.arange(len(y_true)), y_idx].T
    return -np.log(np.clip(true_probs, EPS, 1.0))


def pseudo_labels_from_losses(losses, mode):
    if mode == "raw_loss":
        adjusted = losses
    elif mode == "centered_loss":
        adjusted = losses - losses.mean(axis=0, keepdims=True)
    elif mode == "zscore_loss":
        adjusted = (losses - losses.mean(axis=0, keepdims=True)) / (losses.std(axis=0, keepdims=True) + EPS)
    else:
        raise ValueError(f"Unknown pseudo-label mode: {mode}")
    return adjusted.argmin(axis=1).astype(np.int64)


def assignment_counts_payload(assignments, member_names):
    counts = pd.Series(assignments).value_counts().sort_index()
    return {member_names[idx]: int(counts.get(idx, 0)) for idx in range(len(member_names))}


def assignment_share_payload(assignments, member_names):
    counts = pd.Series(assignments).value_counts().sort_index()
    total = max(len(assignments), 1)
    return {member_names[idx]: float(counts.get(idx, 0) / total) for idx in range(len(member_names))}


def make_member_plan(quick):
    classic = load_json(METRICS_DIR / "classic_boosting_best_params.json")["models"]
    bagging = load_json(METRICS_DIR / "bagging_tree_best_params.json")["models"]
    hist = load_json(METRICS_DIR / "hist_column_sampling_best_params.json")["models"]
    catboost_params = load_json(METRICS_DIR / "catboost_tuned_best_params.json")
    lightgbm_params = load_json(METRICS_DIR / "lightgbm_tuned_best_params.json")
    lightgbm_hist_params = {
        **lightgbm_params,
        "max_bin": hist["lightgbm"]["params"]["max_bin"],
        "colsample_bytree": hist["lightgbm"]["params"]["column_fraction"],
    }
    xgboost_hist_params = hist["xgboost_hist"]["params"]

    members = [
        ("catboost_tuned", "catboost", catboost_params),
        ("lightgbm_hist_column_best", "lightgbm", lightgbm_hist_params),
        ("lightgbm_tuned", "lightgbm", lightgbm_params),
        ("gbdt_log_loss", "sklearn", classic["gbdt_log_loss"]["best_params"]),
        ("hist_gradient_boosting", "sklearn", classic["hist_gradient_boosting"]["best_params"]),
        ("extra_trees", "sklearn", bagging["extra_trees"]["best_params"]),
        ("bagging_decision_tree", "sklearn", bagging["bagging_decision_tree"]["best_params"]),
        ("xgboost_hist_column_best", "xgboost", xgboost_hist_params),
    ]
    if quick:
        return [m for m in members if m[0] in {"catboost_tuned", "lightgbm_hist_column_best", "gbdt_log_loss"}]
    return members


def build_member_probabilities(train_df, test_df, cv, numeric_features, categorical_features, n_jobs, quick):
    member_names = []
    oof_probas = []
    test_probas = []
    member_metrics = []
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    for name, kind, params in make_member_plan(quick):
        if kind == "catboost":
            oof, test = machine_only.catboost_oof_test(params, train_df, test_df, cv)
        elif kind == "lightgbm":
            oof, test = machine_only.lightgbm_oof_test(name, params, train_df, test_df, cv, n_jobs)
        elif kind == "xgboost":
            oof, test = machine_only.xgboost_oof_test(params, train_df, test_df, cv, n_jobs)
        else:
            oof, test = machine_only.sklearn_oof_test(
                name,
                params,
                train_df,
                test_df,
                cv,
                numeric_features,
                categorical_features,
            )
        member_names.append(name)
        oof_probas.append(oof)
        test_probas.append(test)
        member_metrics.append(evaluate(f"{name}_test", y_test, test))
        oof_pred = np.array([LABELS[idx] for idx in oof.argmax(axis=1)])
        member_metrics.append({
            "model": f"{name}_oof",
            "f1_macro": f1_score(y_train, oof_pred, average="macro", zero_division=0),
        })
    return member_names, oof_probas, test_probas, member_metrics


def route_with_assignments(assignments, test_probas):
    output = np.zeros_like(test_probas[0])
    for member_idx, proba in enumerate(test_probas):
        mask = assignments == member_idx
        if mask.any():
            output[mask] = proba[mask]
    return output


def router_cv(y_router, requested_splits):
    counts = pd.Series(y_router).value_counts()
    if len(counts) < 2:
        return None
    min_count = int(counts.min())
    if min_count >= 2:
        return StratifiedKFold(n_splits=min(requested_splits, min_count), shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=min(requested_splits, len(y_router)), shuffle=True, random_state=RANDOM_STATE)


def router_oof_predictions(router_name, X, y_router, n_members, requested_splits, numeric_features, categorical_features, n_jobs):
    cv = router_cv(y_router, requested_splits)
    if cv is None:
        only_member = int(y_router[0])
        hard = np.full(len(X), only_member, dtype=np.int64)
        soft = np.zeros((len(X), n_members), dtype=float)
        soft[:, only_member] = 1.0
        return hard, soft
    hard = np.zeros(len(X), dtype=np.int64)
    soft = np.zeros((len(X), n_members), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y_router), start=1):
        print(f"[ROUTER-OOF] router={router_name} fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        router = make_router(router_name, numeric_features, categorical_features, RANDOM_STATE + 2000 + fold_idx, n_jobs)
        router.fit(X.iloc[train_idx], y_router[train_idx])
        hard[valid_idx] = router.predict(X.iloc[valid_idx])
        soft[valid_idx] = align_router_proba(router.predict_proba(X.iloc[valid_idx]), router.classes_, n_members)
    return hard, soft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--quick", action="store_true", help="Use three experts and one router for smoke testing.")
    parser.add_argument("--output-prefix", default="supervised_error_router_moe")
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    member_names, oof_probas, test_probas, member_metrics = build_member_probabilities(
        train_df,
        test_df,
        cv,
        numeric_features,
        categorical_features,
        args.n_jobs,
        args.quick,
    )
    losses = true_class_losses(oof_probas, y_train)

    metric_records = []
    metric_records.extend(member_metrics)

    routers = ["logistic_router"] if args.quick else ["logistic_router", "extra_trees_router", "random_forest_router"]
    pseudo_label_modes = ["raw_loss"] if args.quick else ["raw_loss", "centered_loss", "zscore_loss"]
    stacked_oof = np.stack(oof_probas, axis=1)
    stacked_test = np.stack(test_probas, axis=1)
    router_records = []
    for pseudo_label_mode in pseudo_label_modes:
        best_member_idx = pseudo_labels_from_losses(losses, pseudo_label_mode)
        member_count_payload = assignment_counts_payload(best_member_idx, member_names)
        member_share_payload = assignment_share_payload(best_member_idx, member_names)

        oracle_oof_hard = np.zeros_like(oof_probas[0])
        for member_idx in range(len(member_names)):
            mask = best_member_idx == member_idx
            if mask.any():
                oracle_oof_hard[mask] = oof_probas[member_idx][mask]
        oracle_row = evaluate(f"supervised_router_oracle_oof_upper_bound_{pseudo_label_mode}", y_train, oracle_oof_hard)
        oracle_row.update({
            "pseudo_label_mode": pseudo_label_mode,
            "pseudo_label_counts_json": json.dumps(member_count_payload, ensure_ascii=False, sort_keys=True),
            "pseudo_label_shares_json": json.dumps(member_share_payload, ensure_ascii=False, sort_keys=True),
        })
        metric_records.append(oracle_row)

        for router_name in routers:
            print(f"[ROUTER] mode={pseudo_label_mode} fitting {router_name}", flush=True)
            router_oof_hard, router_oof_soft = router_oof_predictions(
                router_name,
                X_train,
                best_member_idx,
                len(member_names),
                args.cv,
                numeric_features,
                categorical_features,
                args.n_jobs,
            )
            oof_hard_proba = np.zeros_like(oof_probas[0])
            for member_idx in range(len(member_names)):
                mask = router_oof_hard == member_idx
                if mask.any():
                    oof_hard_proba[mask] = oof_probas[member_idx][mask]
            oof_soft_proba = (stacked_oof * router_oof_soft[:, :, None]).sum(axis=1)

            router = make_router(router_name, numeric_features, categorical_features, RANDOM_STATE, args.n_jobs)
            router.fit(X_train, best_member_idx)
            test_hard_assign = router.predict(X_test)
            test_router_soft = align_router_proba(router.predict_proba(X_test), router.classes_, len(member_names))
            test_hard_proba = route_with_assignments(test_hard_assign, test_probas)
            test_soft_proba = (stacked_test * test_router_soft[:, :, None]).sum(axis=1)
            test_route_share_payload = assignment_share_payload(test_hard_assign, member_names)

            for split, routing, proba in [
                ("oof", "hard", oof_hard_proba),
                ("oof", "soft", oof_soft_proba),
                ("test", "hard", test_hard_proba),
                ("test", "soft", test_soft_proba),
            ]:
                y_true = y_train if split == "oof" else y_test
                row = evaluate(
                    f"supervised_error_router_moe_{pseudo_label_mode}_{router_name}_{routing}_{split}",
                    y_true,
                    proba,
                )
                row.update({
                    "pseudo_label_mode": pseudo_label_mode,
                    "router": router_name,
                    "routing": routing,
                    "split": split,
                    "members_json": json.dumps(member_names, ensure_ascii=False),
                    "pseudo_label_counts_json": json.dumps(member_count_payload, ensure_ascii=False, sort_keys=True),
                    "pseudo_label_shares_json": json.dumps(member_share_payload, ensure_ascii=False, sort_keys=True),
                    "test_hard_route_shares_json": json.dumps(test_route_share_payload, ensure_ascii=False, sort_keys=True),
                })
                metric_records.append(row)
                router_records.append(row)

    metrics_df = pd.DataFrame(metric_records)
    test_router_df = pd.DataFrame(router_records)
    test_router_df = test_router_df[test_router_df["split"] == "test"].sort_values("f1_macro", ascending=False)
    best = test_router_df.iloc[0].to_dict()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_table(metrics_df.to_dict(orient="records"), METRICS_DIR / f"{args.output_prefix}_metrics.csv")
    (METRICS_DIR / f"{args.output_prefix}_best.json").write_text(
        json.dumps(json_safe(best), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(test_router_df[[
        "model",
        "accuracy",
        "f1_macro",
        "roc_auc_ovr_macro",
        "fmi",
        "dropout_accuracy",
        "enrolled_accuracy",
        "graduate_accuracy",
    ]].to_string(index=False))
    print(f"saved metrics: {METRICS_DIR / f'{args.output_prefix}_metrics.csv'}")
    print(f"saved best: {METRICS_DIR / f'{args.output_prefix}_best.json'}")


if __name__ == "__main__":
    main()
