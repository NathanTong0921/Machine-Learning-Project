"""Build an OOF probability ensemble scaffold.

Declaration:
- This script tunes ensemble weights only on out-of-fold training predictions.
- It then refits available members on the full training split and evaluates the
  fixed-weight ensemble on the held-out test split.
- Current members: tuned CatBoost, tuned LightGBM, best MultiViewGatedMLP,
  optional TabPFN.
"""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_models import train_multiview_gated_mlp
from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table


CV_SPLITS = 5
WEIGHT_STEP = 0.05
LIGHTGBM_N_JOBS = int(os.environ.get("LIGHTGBM_N_JOBS", "108"))
BEST_MV_CONFIG = {
    "seed": 62,
    "hidden_dim": 64,
    "dropout": 0.10,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 128,
}


def load_catboost_params():
    path = METRICS_DIR / "catboost_tuned_best_params.json"
    if not path.exists():
        raise FileNotFoundError("Missing catboost_tuned_best_params.json.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lightgbm_params():
    path = METRICS_DIR / "lightgbm_tuned_best_params.json"
    if not path.exists():
        raise FileNotFoundError("Missing lightgbm_tuned_best_params.json.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_catboost_xy(df):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    cat_cols = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    cat_idx = [X.columns.get_loc(col) for col in cat_cols]
    for col in cat_cols:
        X[col] = X[col].astype(str)
    return X, y, cat_idx


def prepare_lightgbm_xy(df):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    cat_cols = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    for col in cat_cols:
        X[col] = X[col].astype("category")
    return X, y, cat_cols


def align_proba(proba, classes):
    aligned = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    class_to_col = {str(label): idx for idx, label in enumerate(classes)}
    for out_idx, label in enumerate(LABELS):
        aligned[:, out_idx] = proba[:, class_to_col[label]]
    return aligned


def make_catboost(params, seed=RANDOM_STATE):
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=seed,
        thread_count=-1,
        **params,
    )


def catboost_oof_and_test(train_df, test_df):
    params = load_catboost_params()
    X, y, cat_idx = prepare_catboost_xy(train_df)
    X_test, _, _ = prepare_catboost_xy(test_df)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member=catboost_tuned fold={fold_idx}/{CV_SPLITS}", flush=True)
        model = make_catboost(params, seed=RANDOM_STATE + fold_idx)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_idx)
        oof[valid_idx] = align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)

    model = make_catboost(params)
    model.fit(X, y, cat_features=cat_idx)
    test_proba = align_proba(model.predict_proba(X_test), model.classes_)
    return oof, test_proba


def lightgbm_oof_and_test(train_df, test_df):
    params = load_lightgbm_params()
    X, y, cat_cols = prepare_lightgbm_xy(train_df)
    X_test, _, _ = prepare_lightgbm_xy(test_df)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member=lightgbm_tuned fold={fold_idx}/{CV_SPLITS}", flush=True)
        model = LGBMClassifier(
            objective="multiclass",
            class_weight="balanced",
            random_state=RANDOM_STATE + fold_idx,
            verbosity=-1,
            n_jobs=LIGHTGBM_N_JOBS,
            **params,
        )
        model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
            categorical_feature=cat_cols,
        )
        oof[valid_idx] = align_proba(model.predict_proba(X.iloc[valid_idx]), model.classes_)

    model = LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
        n_jobs=LIGHTGBM_N_JOBS,
        **params,
    )
    model.fit(X, y, categorical_feature=cat_cols)
    test_proba = align_proba(model.predict_proba(X_test), model.classes_)
    return oof, test_proba


def multiview_oof_and_test(train_df, test_df):
    y = train_df[TARGET_COL]
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    test_fold_probs = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(train_df, y), start=1):
        print(f"[PROGRESS] member=multiview_gated_mlp_best fold={fold_idx}/{CV_SPLITS}", flush=True)
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_valid_df = train_df.iloc[valid_idx].reset_index(drop=True)
        config = {**BEST_MV_CONFIG, "seed": BEST_MV_CONFIG["seed"] + fold_idx}
        result = train_multiview_gated_mlp(fold_train_df, fold_valid_df, **config)
        oof[valid_idx] = result["test_prob"]

        test_result = train_multiview_gated_mlp(fold_train_df, test_df, **config)
        test_fold_probs.append(test_result["test_prob"])

    # Average fold-trained models for the test side to avoid using a different model family in CV vs test.
    test_proba = np.mean(test_fold_probs, axis=0)
    return oof, test_proba


def tabpfn_oof_and_test(train_df, test_df):
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    try:
        from tabpfn import TabPFNClassifier
    except Exception as exc:
        print(f"[WARN] skipping TabPFN: import failed: {exc}", flush=True)
        return None, None

    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        print(f"[PROGRESS] member=tabpfn fold={fold_idx}/{CV_SPLITS}", flush=True)
        clf = TabPFNClassifier()
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = align_proba(clf.predict_proba(X.iloc[valid_idx]), clf.classes_)

    clf = TabPFNClassifier()
    clf.fit(X, y)
    test_proba = align_proba(clf.predict_proba(X_test), clf.classes_)
    return oof, test_proba


def evaluate(name, y_true, proba):
    pred = np.array([LABELS[idx] for idx in proba.argmax(axis=1)])
    metrics = classification_metrics(y_true, pred, y_prob=proba, labels=LABELS)
    report = classification_report(y_true, pred, labels=LABELS, output_dict=True, zero_division=0)
    metrics["model"] = name
    for label in LABELS:
        metrics[f"{label.lower()}_precision"] = report[label]["precision"]
        metrics[f"{label.lower()}_recall"] = report[label]["recall"]
        metrics[f"{label.lower()}_f1"] = report[label]["f1-score"]
    return metrics


def generate_weight_grid(n_members):
    if n_members == 1:
        yield np.array([1.0])
        return

    units = int(round(1.0 / WEIGHT_STEP))

    def recurse(prefix, remaining_units, remaining_members):
        if remaining_members == 1:
            yield prefix + [remaining_units]
            return
        for value in range(remaining_units + 1):
            yield from recurse(prefix + [value], remaining_units - value, remaining_members - 1)

    for allocation in recurse([], units, n_members):
        yield np.array(allocation, dtype=float) / units


def search_weights(member_names, oof_probas, y_true):
    best_score = -np.inf
    best_weights = None
    records = []
    stacked = np.stack(oof_probas, axis=0)

    for weights in generate_weight_grid(len(member_names)):
        if np.count_nonzero(weights) == 0:
            continue
        ensemble_proba = np.tensordot(weights, stacked, axes=(0, 0))
        pred = np.array([LABELS[idx] for idx in ensemble_proba.argmax(axis=1)])
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        record = {f"weight_{name}": float(weight) for name, weight in zip(member_names, weights)}
        record["oof_f1_macro"] = float(score)
        records.append(record)
        if score > best_score:
            best_score = score
            best_weights = weights

    grid_df = pd.DataFrame(records).sort_values("oof_f1_macro", ascending=False)
    return best_weights, best_score, grid_df


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    member_outputs = []
    for name, builder in [
        ("catboost_tuned", catboost_oof_and_test),
        ("lightgbm_tuned", lightgbm_oof_and_test),
        ("multiview_gated_mlp_best", multiview_oof_and_test),
        ("tabpfn", tabpfn_oof_and_test),
    ]:
        oof_proba, test_proba = builder(train_df, test_df)
        if oof_proba is None:
            continue
        member_outputs.append((name, oof_proba, test_proba))

    member_names = [item[0] for item in member_outputs]
    oof_probas = [item[1] for item in member_outputs]
    test_probas = [item[2] for item in member_outputs]

    member_metrics = []
    for name, oof_proba, test_proba in member_outputs:
        member_metrics.append(evaluate(f"{name}_oof", y_train, oof_proba))
        member_metrics.append(evaluate(f"{name}_test", y_test, test_proba))

    weights, best_score, grid_df = search_weights(member_names, oof_probas, y_train)
    ensemble_oof = np.tensordot(weights, np.stack(oof_probas, axis=0), axes=(0, 0))
    ensemble_test = np.tensordot(weights, np.stack(test_probas, axis=0), axes=(0, 0))

    ensemble_metrics = [
        evaluate("oof_probability_ensemble_oof", y_train, ensemble_oof),
        evaluate("oof_probability_ensemble_test", y_test, ensemble_test),
    ]
    all_metrics = member_metrics + ensemble_metrics

    metrics_path = METRICS_DIR / "oof_probability_ensemble_metrics.csv"
    save_metrics_table(all_metrics, metrics_path)

    grid_path = METRICS_DIR / "oof_probability_ensemble_weight_grid.csv"
    save_metrics_table(grid_df.to_dict(orient="records"), grid_path)

    weights_payload = {
        "members": member_names,
        "weights": {name: float(weight) for name, weight in zip(member_names, weights)},
        "oof_f1_macro": float(best_score),
        "note": f"Weight grid step={WEIGHT_STEP}; LightGBM n_jobs={LIGHTGBM_N_JOBS}.",
    }
    weights_path = METRICS_DIR / "oof_probability_ensemble_best_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights_payload, f, ensure_ascii=False, indent=2)

    proba_cache_path = METRICS_DIR / "oof_probability_ensemble_probas.npz"
    np.savez_compressed(
        proba_cache_path,
        member_names=np.array(member_names),
        labels=np.array(LABELS),
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        weights=weights,
        oof_probas=np.stack(oof_probas, axis=0),
        test_probas=np.stack(test_probas, axis=0),
        ensemble_oof=ensemble_oof,
        ensemble_test=ensemble_test,
    )

    print(pd.DataFrame(all_metrics).to_string(index=False))
    print(f"best weights: {weights_payload}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved grid: {grid_path}")
    print(f"saved best weights: {weights_path}")
    print(f"saved probability cache: {proba_cache_path}")


if __name__ == "__main__":
    main()
