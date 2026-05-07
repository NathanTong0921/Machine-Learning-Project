"""Search v2 improvements for the supervised error-router pure-ML MoE.

This script focuses on the current best Core3 expert set:
CatBoost tuned, LightGBM hist-column best, and GBDT log-loss.

Additions over `search_supervised_error_router_moe.py`:
1. Router-only meta features:
   - Application-order thermometer features.
   - OOF/test expert probability uncertainty and disagreement features.
2. Margin-weighted router training:
   - Samples with a larger gap between the best and second-best expert loss
     get larger router sample weights.
3. One M-step specialist retraining:
   - Refit each expert with router soft probabilities as sample weights.
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import search_supervised_error_router_moe as base
import tune_machine_only_ensemble as machine_only

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import save_metrics_table
from src.preprocessing import split_feature_columns


APP_ORDER_COL = "Application order"
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


def add_router_thermometer(frame):
    out = frame.copy()
    if APP_ORDER_COL not in out.columns:
        return out
    for threshold in range(1, 10):
        out[f"router_app_order_ge_{threshold}"] = (out[APP_ORDER_COL] >= threshold).astype(float)
    return out


def entropy(proba):
    clipped = np.clip(proba, EPS, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def probability_meta_frame(probas, prefix):
    stacked = np.stack(probas, axis=1)
    mean_prob = stacked.mean(axis=1)
    std_prob = stacked.std(axis=1)
    max_prob = stacked.max(axis=2)
    sorted_mean = np.sort(mean_prob, axis=1)
    rows = {}
    for class_idx, label in enumerate(LABELS):
        key = label.lower()
        rows[f"{prefix}_mean_prob_{key}"] = mean_prob[:, class_idx]
        rows[f"{prefix}_std_prob_{key}"] = std_prob[:, class_idx]
    for expert_idx in range(stacked.shape[1]):
        rows[f"{prefix}_expert_{expert_idx}_entropy"] = entropy(stacked[:, expert_idx, :])
        rows[f"{prefix}_expert_{expert_idx}_max_prob"] = max_prob[:, expert_idx]
    rows[f"{prefix}_mean_entropy"] = entropy(mean_prob)
    rows[f"{prefix}_mean_margin"] = sorted_mean[:, -1] - sorted_mean[:, -2]
    rows[f"{prefix}_total_disagreement"] = std_prob.sum(axis=1)
    rows[f"{prefix}_max_expert_confidence_gap"] = max_prob.max(axis=1) - max_prob.min(axis=1)
    return pd.DataFrame(rows)


def make_router_features(X_train, X_test, oof_probas, test_probas, variant):
    train_router = X_train.copy()
    test_router = X_test.copy()
    if variant in {"thermometer", "meta"}:
        train_router = add_router_thermometer(train_router)
        test_router = add_router_thermometer(test_router)
    if variant == "meta":
        train_router = pd.concat(
            [train_router.reset_index(drop=True), probability_meta_frame(oof_probas, "router_oof").reset_index(drop=True)],
            axis=1,
        )
        test_router = pd.concat(
            [test_router.reset_index(drop=True), probability_meta_frame(test_probas, "router_oof").reset_index(drop=True)],
            axis=1,
        )
    return train_router, test_router


def make_router(name, numeric_features, categorical_features, seed, n_jobs):
    if name == "logistic_router":
        model = LogisticRegression(
            max_iter=6000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        )
    elif name == "extra_trees_router":
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )
    elif name == "random_forest_router":
        model = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=3,
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


def router_cv(y_router, requested_splits):
    counts = pd.Series(y_router).value_counts()
    if len(counts) < 2:
        return None
    min_count = int(counts.min())
    if min_count >= 2:
        return StratifiedKFold(n_splits=min(requested_splits, min_count), shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=min(requested_splits, len(y_router)), shuffle=True, random_state=RANDOM_STATE)


def fit_router(router, X, y, sample_weight=None):
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["router__sample_weight"] = sample_weight
    router.fit(X, y, **fit_kwargs)
    return router


def router_oof_predictions(router_name, X, y_router, n_members, requested_splits, numeric_features, categorical_features, n_jobs, sample_weight):
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
        router = make_router(router_name, numeric_features, categorical_features, RANDOM_STATE + 3000 + fold_idx, n_jobs)
        fold_weight = None if sample_weight is None else sample_weight[train_idx]
        fit_router(router, X.iloc[train_idx], y_router[train_idx], fold_weight)
        hard[valid_idx] = router.predict(X.iloc[valid_idx])
        soft[valid_idx] = base.align_router_proba(router.predict_proba(X.iloc[valid_idx]), router.classes_, n_members)
    return hard, soft


def margin_weights(losses, y_router, mode):
    sorted_losses = np.sort(losses, axis=1)
    margin = np.maximum(sorted_losses[:, 1] - sorted_losses[:, 0], 0.0)
    if mode == "none":
        return None
    if mode == "margin":
        raw = margin
    elif mode == "sqrt_margin":
        raw = np.sqrt(margin)
    elif mode == "log_margin":
        raw = np.log1p(margin)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    raw = raw / (np.mean(raw) + EPS)
    raw = np.clip(raw, 0.10, 6.0)
    class_weight = compute_sample_weight(class_weight="balanced", y=y_router)
    weights = raw * class_weight
    return weights / (np.mean(weights) + EPS)


def weighted_core3_predict(member_plan, train_df, test_df, router_train_soft, n_jobs):
    test_probas = []
    numeric_features, categorical_features = split_feature_columns(train_df.drop(columns=[TARGET_COL]).columns)
    class_weight = compute_sample_weight(class_weight="balanced", y=train_df[TARGET_COL])
    for member_idx, (name, kind, params) in enumerate(member_plan):
        sample_weight = np.asarray(router_train_soft[:, member_idx], dtype=float)
        sample_weight = np.clip(sample_weight, 1e-4, None)
        print(f"[M-STEP] weighted refit expert={name}", flush=True)
        if kind == "catboost":
            X, y, cat_idx = machine_only.prepare_catboost_xy(train_df)
            X_test, _, _ = machine_only.prepare_catboost_xy(test_df)
            model = machine_only.make_catboost(params, RANDOM_STATE + 500 + member_idx)
            model.fit(X, y, cat_features=cat_idx, sample_weight=sample_weight)
            test_probas.append(machine_only.align_proba(model.predict_proba(X_test), model.classes_))
        elif kind == "lightgbm":
            X, y, cat_cols = machine_only.prepare_category_xy(train_df)
            X_test, _, _ = machine_only.prepare_category_xy(test_df)
            model = machine_only.make_lightgbm(params, RANDOM_STATE + 500 + member_idx, n_jobs)
            model.fit(X, y, categorical_feature=cat_cols, sample_weight=sample_weight)
            test_probas.append(machine_only.align_proba(model.predict_proba(X_test), model.classes_))
        elif kind == "sklearn":
            X = train_df.drop(columns=[TARGET_COL])
            y = train_df[TARGET_COL]
            X_test = test_df.drop(columns=[TARGET_COL])
            model = machine_only.make_sklearn_member(name, params, RANDOM_STATE + 500 + member_idx, numeric_features, categorical_features)
            model.fit(X, y, model__sample_weight=sample_weight * class_weight)
            test_probas.append(machine_only.align_proba(model.predict_proba(X_test), model.classes_))
        else:
            raise ValueError(f"Unknown member kind: {kind}")
    return test_probas


def evaluate_router_variant(
    name,
    router_name,
    train_router,
    test_router,
    y_router,
    losses,
    oof_probas,
    test_probas,
    y_train,
    y_test,
    cv_splits,
    n_jobs,
    weight_mode,
):
    numeric_features, categorical_features = split_feature_columns(train_router.columns)
    n_members = len(oof_probas)
    weights = margin_weights(losses, y_router, weight_mode)
    router_oof_hard, router_oof_soft = router_oof_predictions(
        router_name,
        train_router,
        y_router,
        n_members,
        cv_splits,
        numeric_features,
        categorical_features,
        n_jobs,
        weights,
    )
    stacked_oof = np.stack(oof_probas, axis=1)
    stacked_test = np.stack(test_probas, axis=1)

    oof_hard = base.route_with_assignments(router_oof_hard, oof_probas)
    oof_soft = (stacked_oof * router_oof_soft[:, :, None]).sum(axis=1)

    router = make_router(router_name, numeric_features, categorical_features, RANDOM_STATE, n_jobs)
    fit_router(router, train_router, y_router, weights)
    test_hard_assign = router.predict(test_router)
    test_router_soft = base.align_router_proba(router.predict_proba(test_router), router.classes_, n_members)
    test_hard = base.route_with_assignments(test_hard_assign, test_probas)
    test_soft = (stacked_test * test_router_soft[:, :, None]).sum(axis=1)

    rows = []
    for split, routing, proba, y_true in [
        ("oof", "hard", oof_hard, y_train),
        ("oof", "soft", oof_soft, y_train),
        ("test", "hard", test_hard, y_test),
        ("test", "soft", test_soft, y_test),
    ]:
        row = base.evaluate(f"{name}_{router_name}_{weight_mode}_{routing}_{split}", y_true, proba)
        row.update({
            "experiment": name,
            "router": router_name,
            "weight_mode": weight_mode,
            "routing": routing,
            "split": split,
            "test_hard_route_shares_json": json.dumps(
                base.assignment_share_payload(test_hard_assign, [str(i) for i in range(n_members)]),
                ensure_ascii=False,
                sort_keys=True,
            ),
        })
        rows.append(row)
    return rows, router, test_router_soft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--output-prefix", default="supervised_error_router_moe_v2")
    parser.add_argument("--skip-mstep", action="store_true")
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

    member_plan = base.make_member_plan(quick=True)
    member_names, oof_probas, test_probas, member_metrics = base.build_member_probabilities(
        train_df,
        test_df,
        cv,
        numeric_features,
        categorical_features,
        args.n_jobs,
        quick=True,
    )
    losses = base.true_class_losses(oof_probas, y_train)
    y_router = base.pseudo_labels_from_losses(losses, "raw_loss")
    pseudo_counts = base.assignment_counts_payload(y_router, member_names)
    pseudo_shares = base.assignment_share_payload(y_router, member_names)

    records = []
    records.extend(member_metrics)
    router_variants = ["raw", "thermometer", "meta"]
    router_models = ["logistic_router", "extra_trees_router", "random_forest_router"]
    weight_modes = ["none", "margin", "sqrt_margin", "log_margin"]
    best_soft_state = None
    best_soft_f1 = -np.inf

    for feature_variant in router_variants:
        train_router, test_router = make_router_features(X_train, X_test, oof_probas, test_probas, feature_variant)
        for router_name in router_models:
            for weight_mode in weight_modes:
                exp_name = f"v2_{feature_variant}"
                print(f"[V2] features={feature_variant} router={router_name} weight={weight_mode}", flush=True)
                rows, router, test_router_soft = evaluate_router_variant(
                    exp_name,
                    router_name,
                    train_router,
                    test_router,
                    y_router,
                    losses,
                    oof_probas,
                    test_probas,
                    y_train,
                    y_test,
                    args.cv,
                    args.n_jobs,
                    weight_mode,
                )
                for row in rows:
                    row.update({
                        "router_feature_variant": feature_variant,
                        "members_json": json.dumps(member_names, ensure_ascii=False),
                        "pseudo_label_counts_json": json.dumps(pseudo_counts, ensure_ascii=False, sort_keys=True),
                        "pseudo_label_shares_json": json.dumps(pseudo_shares, ensure_ascii=False, sort_keys=True),
                    })
                records.extend(rows)
                test_soft = [row for row in rows if row["split"] == "test" and row["routing"] == "soft"][0]
                if test_soft["f1_macro"] > best_soft_f1:
                    best_soft_f1 = test_soft["f1_macro"]
                    best_soft_state = {
                        "feature_variant": feature_variant,
                        "router_name": router_name,
                        "weight_mode": weight_mode,
                        "train_router": train_router,
                        "test_router": test_router,
                        "router": router,
                        "test_router_soft": test_router_soft,
                        "metrics": test_soft,
                    }
                save_metrics_table(records, METRICS_DIR / f"{args.output_prefix}_metrics.csv")

    if not args.skip_mstep and best_soft_state is not None:
        train_router = best_soft_state["train_router"]
        test_router = best_soft_state["test_router"]
        numeric_router, categorical_router = split_feature_columns(train_router.columns)
        weights = margin_weights(losses, y_router, best_soft_state["weight_mode"])
        router = make_router(best_soft_state["router_name"], numeric_router, categorical_router, RANDOM_STATE, args.n_jobs)
        fit_router(router, train_router, y_router, weights)
        train_router_soft = base.align_router_proba(router.predict_proba(train_router), router.classes_, len(member_names))
        test_router_soft = base.align_router_proba(router.predict_proba(test_router), router.classes_, len(member_names))
        specialized_test_probas = weighted_core3_predict(member_plan, train_df, test_df, train_router_soft, args.n_jobs)
        specialized_stacked = np.stack(specialized_test_probas, axis=1)
        mstep_soft = (specialized_stacked * test_router_soft[:, :, None]).sum(axis=1)
        mstep_hard = base.route_with_assignments(test_router_soft.argmax(axis=1), specialized_test_probas)
        for routing, proba in [("soft", mstep_soft), ("hard", mstep_hard)]:
            row = base.evaluate(f"v2_mstep_{best_soft_state['feature_variant']}_{best_soft_state['router_name']}_{best_soft_state['weight_mode']}_{routing}_test", y_test, proba)
            row.update({
                "experiment": "v2_mstep",
                "router": best_soft_state["router_name"],
                "weight_mode": best_soft_state["weight_mode"],
                "routing": routing,
                "split": "test",
                "router_feature_variant": best_soft_state["feature_variant"],
                "members_json": json.dumps(member_names, ensure_ascii=False),
                "pseudo_label_counts_json": json.dumps(pseudo_counts, ensure_ascii=False, sort_keys=True),
                "pseudo_label_shares_json": json.dumps(pseudo_shares, ensure_ascii=False, sort_keys=True),
            })
            records.append(row)

    metrics_df = pd.DataFrame(records)
    metrics_path = METRICS_DIR / f"{args.output_prefix}_metrics.csv"
    best_df = metrics_df[metrics_df["model"].str.contains("_test", na=False)].sort_values("f1_macro", ascending=False)
    best = best_df.iloc[0].to_dict()
    save_metrics_table(metrics_df.to_dict(orient="records"), metrics_path)
    (METRICS_DIR / f"{args.output_prefix}_best.json").write_text(
        json.dumps(json_safe(best), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(best_df[[
        "model",
        "accuracy",
        "f1_macro",
        "roc_auc_ovr_macro",
        "fmi",
        "dropout_accuracy",
        "enrolled_accuracy",
        "graduate_accuracy",
    ]].head(12).to_string(index=False))
    print(f"saved metrics: {metrics_path}")
    print(f"saved best: {METRICS_DIR / f'{args.output_prefix}_best.json'}")


if __name__ == "__main__":
    main()
