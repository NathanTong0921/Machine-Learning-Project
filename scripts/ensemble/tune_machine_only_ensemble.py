"""Tune an OOF probability ensemble using only classical machine-learning models.

Declaration:
- This script explicitly excludes MLP, Transformer, TabPFN, and any other deep
  learning / foundation-model member.
- Members are restricted to bagging, extra trees, classic boosting, and
  industrial tree boosting models.
- Ensemble weights are selected only on out-of-fold training probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

if not hasattr(np, "NaN"):
    np.NaN = np.nan

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FIGURE_DIR, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import split_feature_columns
from src.visualization import save_metrics_table_figure


WEIGHT_STEP = 0.05
REFINED_WEIGHT_STEP = 0.025


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


def align_proba(proba, classes):
    aligned = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    class_to_col = {str(label): idx for idx, label in enumerate(classes)}
    for out_idx, label in enumerate(LABELS):
        aligned[:, out_idx] = proba[:, class_to_col[label]]
    return aligned


def dense_preprocessor(numeric_features, categorical_features):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def strip_model_prefix(params):
    return {key.removeprefix("model__"): value for key, value in params.items()}


def prepare_catboost_xy(df):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    cat_cols = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    cat_idx = [X.columns.get_loc(col) for col in cat_cols]
    for col in cat_cols:
        X[col] = X[col].astype(str)
    return X, y, cat_idx


def prepare_category_xy(df):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    cat_cols = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    for col in cat_cols:
        X[col] = X[col].astype("category")
    return X, y, cat_cols


def make_sklearn_member(name, params, seed, numeric_features, categorical_features):
    if name == "adaboost_stumps":
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=seed), random_state=seed)
    elif name == "gbdt_log_loss":
        model = GradientBoostingClassifier(loss="log_loss", random_state=seed)
    elif name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(loss="log_loss", early_stopping=False, random_state=seed)
    elif name == "bagging_decision_tree":
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
            bootstrap=True,
            random_state=seed,
            n_jobs=1,
        )
    elif name == "pasting_decision_tree":
        model = BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
            bootstrap=False,
            random_state=seed,
            n_jobs=1,
        )
    elif name == "extra_trees":
        model = ExtraTreesClassifier(random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown sklearn member: {name}")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", dense_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )
    pipeline.set_params(**params)
    return pipeline


def make_catboost(params, seed):
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=seed,
        thread_count=-1,
        **params,
    )


def make_lightgbm(params, seed, n_jobs):
    return LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
        n_jobs=n_jobs,
        **params,
    )


def make_xgboost_hist(params, seed, n_jobs):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.05,
        n_estimators=400,
        subsample=0.9,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        max_bin=int(params["max_bin"]),
        colsample_bytree=float(params["column_fraction"]),
    )


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


def sklearn_oof_test(name, params, train_df, test_df, cv, numeric_features, categorical_features):
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    sample_weight = compute_sample_weight(class_weight="balanced", y=y)
    use_sample_weight = name in {"adaboost_stumps", "gbdt_log_loss"}
    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member={name} fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        model = make_sklearn_member(name, params, RANDOM_STATE + fold_idx, numeric_features, categorical_features)
        fit_kwargs = {"model__sample_weight": sample_weight[train_idx]} if use_sample_weight else {}
        model.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
        oof[valid_idx] = align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)
    model = make_sklearn_member(name, params, RANDOM_STATE, numeric_features, categorical_features)
    fit_kwargs = {"model__sample_weight": sample_weight} if use_sample_weight else {}
    model.fit(X, y, **fit_kwargs)
    test_proba = align_proba(model.predict_proba(X_test), model.classes_)
    return oof, test_proba


def catboost_oof_test(params, train_df, test_df, cv):
    X, y, cat_idx = prepare_catboost_xy(train_df)
    X_test, _, _ = prepare_catboost_xy(test_df)
    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member=catboost_tuned fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        model = make_catboost(params, RANDOM_STATE + fold_idx)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_idx)
        oof[valid_idx] = align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)
    model = make_catboost(params, RANDOM_STATE)
    model.fit(X, y, cat_features=cat_idx)
    test_proba = align_proba(model.predict_proba(X_test), model.classes_)
    return oof, test_proba


def lightgbm_oof_test(name, params, train_df, test_df, cv, n_jobs):
    X, y, cat_cols = prepare_category_xy(train_df)
    X_test, _, _ = prepare_category_xy(test_df)
    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member={name} fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        model = make_lightgbm(params, RANDOM_STATE + fold_idx, n_jobs)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], categorical_feature=cat_cols)
        oof[valid_idx] = align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)
    model = make_lightgbm(params, RANDOM_STATE, n_jobs)
    model.fit(X, y, categorical_feature=cat_cols)
    test_proba = align_proba(model.predict_proba(X_test), model.classes_)
    return oof, test_proba


def xgboost_oof_test(params, train_df, test_df, cv, n_jobs):
    X, y_text, _ = prepare_category_xy(train_df)
    X_test, _, _ = prepare_category_xy(test_df)
    encoder = LabelEncoder().fit(LABELS)
    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y_text), start=1):
        print(f"[PROGRESS] member=xgboost_hist fold={fold_idx}/{cv.get_n_splits()}", flush=True)
        model = make_xgboost_hist(params, RANDOM_STATE + fold_idx, n_jobs)
        model.fit(X.iloc[train_idx], encoder.transform(y_text.iloc[train_idx]))
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])
    model = make_xgboost_hist(params, RANDOM_STATE, n_jobs)
    model.fit(X, encoder.transform(y_text))
    test_proba = model.predict_proba(X_test)
    return oof, test_proba


def generate_weight_grid(n_members, step):
    units = int(round(1.0 / step))

    def rec(prefix, remaining, members_left):
        if members_left == 1:
            yield prefix + [remaining]
            return
        for value in range(remaining + 1):
            yield from rec(prefix + [value], remaining - value, members_left - 1)

    for weights in rec([], units, n_members):
        yield np.array(weights, dtype=float) / units


def macro_f1_from_indices(y_true_idx, y_pred_idx, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true_idx, y_pred_idx), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom != 0)
    return float(f1.mean())


def search_weights(member_names, oof_probas, y_true, step):
    records = []
    best_score = -np.inf
    best_weights = None
    stacked = np.stack(oof_probas, axis=0)
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    y_true_idx = np.array([label_to_idx[str(label)] for label in y_true], dtype=np.int64)
    for weights in generate_weight_grid(len(member_names), step):
        proba = np.tensordot(weights, stacked, axes=(0, 0))
        pred_idx = proba.argmax(axis=1)
        score = macro_f1_from_indices(y_true_idx, pred_idx, len(LABELS))
        record = {f"weight_{name}": float(weight) for name, weight in zip(member_names, weights)}
        record["oof_f1_macro"] = float(score)
        records.append(record)
        if score > best_score:
            best_score = score
            best_weights = weights
    return best_weights, best_score, pd.DataFrame(records).sort_values("oof_f1_macro", ascending=False)


def refine_weights(member_names, oof_probas, y_true, coarse_weights):
    return coarse_weights, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--max-ensemble-members", type=int, default=6)
    parser.add_argument("--quick", action="store_true", help="Use a smaller member set for smoke testing.")
    args = parser.parse_args()

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    numeric_features, categorical_features = split_feature_columns(train_df.drop(columns=[TARGET_COL]).columns)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)

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

    member_names = []
    oof_probas = []
    test_probas = []

    members = [
        ("catboost_tuned", "catboost", catboost_params),
        ("lightgbm_tuned", "lightgbm", lightgbm_params),
        ("lightgbm_hist_column_best", "lightgbm", lightgbm_hist_params),
        ("gbdt_log_loss", "sklearn", classic["gbdt_log_loss"]["best_params"]),
        ("hist_gradient_boosting", "sklearn", classic["hist_gradient_boosting"]["best_params"]),
        ("adaboost_stumps", "sklearn", classic["adaboost_stumps"]["best_params"]),
        ("pasting_decision_tree", "sklearn", bagging["pasting_decision_tree"]["best_params"]),
        ("extra_trees", "sklearn", bagging["extra_trees"]["best_params"]),
        ("bagging_decision_tree", "sklearn", bagging["bagging_decision_tree"]["best_params"]),
        ("xgboost_hist_column_best", "xgboost", xgboost_hist_params),
    ]
    if args.quick:
        members = [m for m in members if m[0] in {"catboost_tuned", "lightgbm_hist_column_best", "gbdt_log_loss"}]

    metric_records = []
    member_oof_scores = []
    for name, kind, params in members:
        if kind == "catboost":
            oof, test = catboost_oof_test(params, train_df, test_df, cv)
        elif kind == "lightgbm":
            oof, test = lightgbm_oof_test(name, params, train_df, test_df, cv, args.n_jobs)
        elif kind == "xgboost":
            oof, test = xgboost_oof_test(params, train_df, test_df, cv, args.n_jobs)
        else:
            oof, test = sklearn_oof_test(name, params, train_df, test_df, cv, numeric_features, categorical_features)
        member_names.append(name)
        oof_probas.append(oof)
        test_probas.append(test)
        metric_records.append(evaluate(f"{name}_test", y_test, test))
        oof_pred = np.array([LABELS[idx] for idx in oof.argmax(axis=1)])
        member_oof_scores.append(
            {
                "member": name,
                "oof_f1_macro": f1_score(y_train, oof_pred, average="macro", zero_division=0),
            }
        )

    selected_members_df = pd.DataFrame(member_oof_scores).sort_values("oof_f1_macro", ascending=False)
    if args.quick:
        max_members = len(member_names)
    else:
        max_members = min(args.max_ensemble_members, len(member_names))
    selected_names = selected_members_df.head(max_members)["member"].tolist()
    selected_idx = [member_names.index(name) for name in selected_names]
    selected_oof = [oof_probas[idx] for idx in selected_idx]
    selected_test = [test_probas[idx] for idx in selected_idx]

    save_metrics_table(selected_members_df.to_dict(orient="records"), METRICS_DIR / "machine_only_ensemble_member_oof_scores.csv")

    coarse_weights, coarse_score, grid_df = search_weights(selected_names, selected_oof, y_train, WEIGHT_STEP)
    refined_weights, refined_score, refined_grid = refine_weights(selected_names, selected_oof, y_train, coarse_weights)
    if refined_score is not None and refined_score >= coarse_score:
        selected_weights = refined_weights
        best_score = refined_score
    else:
        selected_weights = coarse_weights
        best_score = coarse_score

    weights = np.zeros(len(member_names), dtype=float)
    for idx, weight in zip(selected_idx, selected_weights):
        weights[idx] = weight

    ensemble_oof = np.tensordot(selected_weights, np.stack(selected_oof, axis=0), axes=(0, 0))
    ensemble_test = np.tensordot(selected_weights, np.stack(selected_test, axis=0), axes=(0, 0))
    metric_records.append(evaluate("machine_only_ensemble_oof", y_train, ensemble_oof))
    metric_records.append(evaluate("machine_only_ensemble_test", y_test, ensemble_test))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_table(metric_records, METRICS_DIR / "machine_only_ensemble_metrics.csv")
    save_metrics_table(grid_df.to_dict(orient="records"), METRICS_DIR / "machine_only_ensemble_weight_grid.csv")
    if refined_grid is not None:
        save_metrics_table(refined_grid.to_dict(orient="records"), METRICS_DIR / "machine_only_ensemble_refined_weight_grid.csv")

    best_payload = {
        "members": member_names,
        "selected_members": selected_names,
        "excluded": ["multiview_gated_mlp_best", "all MLP variants", "all transformer variants", "tabpfn"],
        "weights": {name: float(weight) for name, weight in zip(member_names, weights)},
        "oof_f1_macro": float(best_score),
        "cv_splits": args.cv,
        "max_ensemble_members": args.max_ensemble_members,
        "weight_step": WEIGHT_STEP,
        "refined_weight_step": REFINED_WEIGHT_STEP,
        "selection_metric": "OOF macro-F1; held-out test evaluated once",
        "metric_notes": {
            "fmi": "Fowlkes-Mallows index from true and predicted class labels.",
            "class_accuracy": "Per-class accuracy is TP / true class count, equivalent to class recall.",
        },
    }
    (METRICS_DIR / "machine_only_ensemble_best_weights.json").write_text(
        json.dumps(json_safe(best_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_df = pd.DataFrame(metric_records)
    save_metrics_table_figure(
        metrics_df[metrics_df["model"].isin(["machine_only_ensemble_test"])][
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
        FIGURE_DIR / "machine_only_ensemble_summary.png",
        "Machine-Only Ensemble",
    )

    print(pd.DataFrame(metric_records).sort_values("f1_macro", ascending=False).to_string(index=False))
    print(f"weights: {best_payload['weights']}")


if __name__ == "__main__":
    main()
