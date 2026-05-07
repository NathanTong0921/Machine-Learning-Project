"""Try adding semester-feature tree members to the OOF probability ensemble.

Declaration:
- Existing base member probabilities are loaded from oof_probability_ensemble.py.
- Only new semester-feature tree members are trained here.
- Ensemble weights are selected on OOF predictions only.
- The held-out test set is evaluated once with the selected weights.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table


CV_SPLITS = 5
WEIGHT_STEP = 0.05
REFINED_WEIGHT_STEP = 0.025
LIGHTGBM_N_JOBS = 108


def safe_divide(numerator, denominator):
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def add_semester_ratio_features(df):
    df = df.copy()
    first_enrolled = df["Curricular units 1st sem (enrolled)"]
    second_enrolled = df["Curricular units 2nd sem (enrolled)"]
    first_evaluations = df["Curricular units 1st sem (evaluations)"]
    second_evaluations = df["Curricular units 2nd sem (evaluations)"]
    first_approved = df["Curricular units 1st sem (approved)"]
    second_approved = df["Curricular units 2nd sem (approved)"]

    df["first_approved_rate"] = safe_divide(first_approved, first_enrolled)
    df["second_approved_rate"] = safe_divide(second_approved, second_enrolled)
    df["first_eval_pass_rate"] = safe_divide(first_approved, first_evaluations)
    df["second_eval_pass_rate"] = safe_divide(second_approved, second_evaluations)
    df["grade_delta"] = df["Curricular units 2nd sem (grade)"] - df["Curricular units 1st sem (grade)"]
    df["approved_delta"] = second_approved - first_approved
    df["tuition_scholarship_interaction"] = df["Tuition fees up to date"] * df["Scholarship holder"]
    df["debtor_tuition_interaction"] = df["Debtor"] * df["Tuition fees up to date"]
    return df


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_proba(proba, classes):
    aligned = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    class_to_col = {str(label): idx for idx, label in enumerate(classes)}
    for out_idx, label in enumerate(LABELS):
        aligned[:, out_idx] = proba[:, class_to_col[label]]
    return aligned


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


def make_lightgbm(params, seed):
    return LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
        n_jobs=LIGHTGBM_N_JOBS,
        **params,
    )


def build_member_probas(name, train_df, test_df, params, prepare_xy, build_model):
    X_train, y_train, categorical = prepare_xy(train_df)
    X_test, _, _ = prepare_xy(test_df)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    oof = np.zeros((len(train_df), len(LABELS)), dtype=float)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
        print(f"[PROGRESS] member={name} fold={fold_idx}/{CV_SPLITS}", flush=True)
        model = build_model(params, RANDOM_STATE + fold_idx)
        fit_kwargs = {"cat_features": categorical} if name.startswith("catboost") else {"categorical_feature": categorical}
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], **fit_kwargs)
        oof[valid_idx] = align_proba(model.predict_proba(X_train.iloc[valid_idx]), model.classes_)

    model = build_model(params, RANDOM_STATE)
    fit_kwargs = {"cat_features": categorical} if name.startswith("catboost") else {"categorical_feature": categorical}
    model.fit(X_train, y_train, **fit_kwargs)
    test_proba = align_proba(model.predict_proba(X_test), model.classes_)
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


def generate_weight_grid(n_members, weight_step):
    units = int(round(1.0 / weight_step))

    def recurse(prefix, remaining_units, remaining_members):
        if remaining_members == 1:
            yield prefix + [remaining_units]
            return
        for value in range(remaining_units + 1):
            yield from recurse(prefix + [value], remaining_units - value, remaining_members - 1)

    for allocation in recurse([], units, n_members):
        yield np.array(allocation, dtype=float) / units


def search_weights(member_names, oof_probas, y_true, weight_step):
    records = []
    best_score = -np.inf
    best_weights = None
    stacked = np.stack(oof_probas, axis=0)

    for weights in generate_weight_grid(len(member_names), weight_step):
        ensemble_proba = np.tensordot(weights, stacked, axes=(0, 0))
        pred = np.array([LABELS[idx] for idx in ensemble_proba.argmax(axis=1)])
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        record = {f"weight_{name}": float(weight) for name, weight in zip(member_names, weights)}
        record["oof_f1_macro"] = float(score)
        records.append(record)
        if score > best_score:
            best_score = score
            best_weights = weights

    return best_weights, best_score, pd.DataFrame(records).sort_values("oof_f1_macro", ascending=False)


def refine_active_weights(member_names, oof_probas, y_true, coarse_weights):
    active_idx = np.flatnonzero(coarse_weights > 0)
    if len(active_idx) < 2:
        return coarse_weights, None

    active_names = [member_names[idx] for idx in active_idx]
    active_oof_probas = [oof_probas[idx] for idx in active_idx]
    active_weights, _, grid_df = search_weights(active_names, active_oof_probas, y_true, REFINED_WEIGHT_STEP)

    refined_weights = np.zeros_like(coarse_weights)
    for idx, weight in zip(active_idx, active_weights):
        refined_weights[idx] = weight
    return refined_weights, grid_df


def main():
    cache_path = METRICS_DIR / "oof_probability_ensemble_probas.npz"
    if not cache_path.exists():
        raise FileNotFoundError("Run `python scripts/oof_probability_ensemble.py` first.")
    cache = np.load(cache_path, allow_pickle=True)

    base_member_names = cache["member_names"].astype(str).tolist()
    y_train = cache["y_train"].astype(str)
    y_test = cache["y_test"].astype(str)
    oof_probas = [arr for arr in cache["oof_probas"]]
    test_probas = [arr for arr in cache["test_probas"]]
    member_names = list(base_member_names)

    train_df = add_semester_ratio_features(pd.read_csv(PROCESSED_DIR / "train.csv"))
    test_df = add_semester_ratio_features(pd.read_csv(PROCESSED_DIR / "test.csv"))

    for name, params, prepare_xy, build_model in [
        (
            "catboost_tuned_semester",
            load_json(METRICS_DIR / "catboost_tuned_best_params.json"),
            prepare_catboost_xy,
            make_catboost,
        ),
        (
            "lightgbm_tuned_semester",
            load_json(METRICS_DIR / "lightgbm_tuned_best_params.json"),
            prepare_lightgbm_xy,
            make_lightgbm,
        ),
    ]:
        oof, test = build_member_probas(name, train_df, test_df, params, prepare_xy, build_model)
        member_names.append(name)
        oof_probas.append(oof)
        test_probas.append(test)

    coarse_weights, coarse_best_score, grid_df = search_weights(member_names, oof_probas, y_train, WEIGHT_STEP)
    refined_weights, refined_grid_df = refine_active_weights(member_names, oof_probas, y_train, coarse_weights)
    if refined_grid_df is not None:
        refined_oof = np.tensordot(refined_weights, np.stack(oof_probas, axis=0), axes=(0, 0))
        refined_pred = np.array([LABELS[idx] for idx in refined_oof.argmax(axis=1)])
        refined_score = f1_score(y_train, refined_pred, average="macro", zero_division=0)
        if refined_score >= coarse_best_score:
            weights = refined_weights
            best_score = refined_score
        else:
            weights = coarse_weights
            best_score = coarse_best_score
    else:
        weights = coarse_weights
        best_score = coarse_best_score
    ensemble_oof = np.tensordot(weights, np.stack(oof_probas, axis=0), axes=(0, 0))
    ensemble_test = np.tensordot(weights, np.stack(test_probas, axis=0), axes=(0, 0))

    metrics = []
    for name, oof, test in zip(member_names, oof_probas, test_probas):
        if name.endswith("_semester"):
            metrics.append(evaluate(f"{name}_oof", y_train, oof))
            metrics.append(evaluate(f"{name}_test", y_test, test))
    metrics.extend(
        [
            evaluate("oof_probability_ensemble_semester_members_oof", y_train, ensemble_oof),
            evaluate("oof_probability_ensemble_semester_members_test", y_test, ensemble_test),
        ]
    )

    metrics_path = METRICS_DIR / "oof_probability_ensemble_semester_members_metrics.csv"
    save_metrics_table(metrics, metrics_path)

    grid_path = METRICS_DIR / "oof_probability_ensemble_semester_members_weight_grid.csv"
    save_metrics_table(grid_df.to_dict(orient="records"), grid_path)

    refined_grid_path = None
    if refined_grid_df is not None:
        refined_grid_path = METRICS_DIR / "oof_probability_ensemble_semester_members_refined_weight_grid.csv"
        save_metrics_table(refined_grid_df.to_dict(orient="records"), refined_grid_path)

    payload = {
        "members": member_names,
        "weights": {name: float(weight) for name, weight in zip(member_names, weights)},
        "oof_f1_macro": float(best_score),
        "source_probability_cache": str(cache_path),
        "coarse_oof_f1_macro": float(coarse_best_score),
        "coarse_weights": {name: float(weight) for name, weight in zip(member_names, coarse_weights)},
        "note": (
            f"Coarse grid step={WEIGHT_STEP}; refined active-member step={REFINED_WEIGHT_STEP}; "
            "added semester-feature CatBoost/LightGBM candidates."
        ),
    }
    weights_path = METRICS_DIR / "oof_probability_ensemble_semester_members_best_weights.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    proba_cache_path = METRICS_DIR / "oof_probability_ensemble_semester_members_probas.npz"
    np.savez_compressed(
        proba_cache_path,
        member_names=np.array(member_names),
        labels=np.array(LABELS),
        y_train=y_train,
        y_test=y_test,
        weights=weights,
        oof_probas=np.stack(oof_probas, axis=0),
        test_probas=np.stack(test_probas, axis=0),
        ensemble_oof=ensemble_oof,
        ensemble_test=ensemble_test,
    )

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"best weights: {payload}")
    print(f"saved metrics: {metrics_path}")
    print(f"saved grid: {grid_path}")
    if refined_grid_path is not None:
        print(f"saved refined grid: {refined_grid_path}")
    print(f"saved best weights: {weights_path}")
    print(f"saved probability cache: {proba_cache_path}")


if __name__ == "__main__":
    main()
