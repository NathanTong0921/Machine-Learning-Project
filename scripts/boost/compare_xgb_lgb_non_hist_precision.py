"""Compare XGBoost exact splits and high-bin LightGBM against current best tree settings.

Declaration:
- XGBoost can be run with `tree_method="exact"`, but exact mode does not use
  XGBoost's native categorical split path. This script therefore evaluates exact
  XGBoost on one-hot encoded categorical features.
- LightGBM is histogram-based by design. It has no true exact split mode, so the
  closest precision check is to raise `max_bin` while keeping the current best
  tuned structural parameters.

Tips:
- Read macro-F1 first, then class-wise accuracy for Enrolled.
- This script appends a precision-oriented comparison; it does not replace the
  existing histogram search or tuned baseline files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Compatibility shim for older xgboost releases under NumPy 2.x.
if not hasattr(np, "NaN"):
    np.NaN = np.nan

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import classification_metrics, save_metrics_table
from src.preprocessing import build_preprocessor, split_feature_columns


BOOST_METRICS_DIR = METRICS_DIR / "boost"
OUTPUT_METRICS = BOOST_METRICS_DIR / "xgb_lgb_non_hist_precision_metrics.csv"
OUTPUT_METADATA = BOOST_METRICS_DIR / "xgb_lgb_non_hist_precision_metadata.json"
LGB_QUANTILE_OFF_GRID = BOOST_METRICS_DIR / "lightgbm_quantile_off_search_grid.csv"
LGB_QUANTILE_OFF_BEST = BOOST_METRICS_DIR / "lightgbm_quantile_off_best_params.json"
XGB_EXACT_GRID = BOOST_METRICS_DIR / "xgboost_exact_onehot_search_grid.csv"
XGB_EXACT_BEST = BOOST_METRICS_DIR / "xgboost_exact_onehot_best_params.json"


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_data():
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing processed train/test CSV. Run `python scripts/preprocessing/prepare_data.py` first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COL]).copy()
    y_train = train_df[TARGET_COL].copy()
    X_test = test_df.drop(columns=[TARGET_COL]).copy()
    y_test = test_df[TARGET_COL].copy()
    numeric_cols, cat_cols = split_feature_columns(X_train.columns)
    return X_train, y_train, X_test, y_test, numeric_cols, cat_cols


def prepare_native_categorical(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str]):
    X_train_native = X_train.copy()
    X_test_native = X_test.copy()
    for col in cat_cols:
        X_train_native[col] = X_train_native[col].astype("category")
        X_test_native[col] = X_test_native[col].astype("category")
    return X_train_native, X_test_native


def evaluate_text_predictions(model_name: str, y_true, y_pred, y_prob, extra: dict[str, Any]):
    metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, labels=LABELS)
    metrics["model"] = model_name
    metrics.update(extra)
    return metrics


def class_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "dropout_f1": report["Dropout"]["f1-score"],
        "enrolled_precision": report["Enrolled"]["precision"],
        "enrolled_recall": report["Enrolled"]["recall"],
        "enrolled_f1": report["Enrolled"]["f1-score"],
        "graduate_f1": report["Graduate"]["f1-score"],
    }


def run_xgboost_exact_onehot(X_train, y_train, X_test, y_test, numeric_cols, cat_cols, n_jobs: int):
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    params = {
        "objective": "multi:softprob",
        "num_class": len(LABELS),
        "tree_method": "exact",
        "enable_categorical": False,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": n_jobs,
    }
    model = XGBClassifier(**params)
    model.fit(X_train_encoded, y_train_encoded)
    y_pred_encoded = model.predict(X_test_encoded).astype(int)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_prob = model.predict_proba(X_test_encoded)

    return evaluate_text_predictions(
        "xgboost_exact_onehot_best_structural_params",
        y_test,
        y_pred,
        y_prob,
        {
            "split_algorithm": "exact",
            "categorical_handling": "onehot",
            "max_bin": None,
            "colsample_bytree": params["colsample_bytree"],
            "n_input_features": int(X_train_encoded.shape[1]),
        },
    ), params


def xgboost_exact_grid(wide: bool, local: bool):
    if local:
        grid = {
            "max_depth": [5, 6, 7],
            "learning_rate": [0.04, 0.05, 0.06],
            "n_estimators": [350, 400, 450],
            "subsample": [0.9],
            "colsample_bytree": [0.85, 0.9, 0.95],
            "min_child_weight": [1],
            "reg_lambda": [1.0],
        }
    elif wide:
        grid = {
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.02, 0.03, 0.05],
            "n_estimators": [300, 400, 500],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "min_child_weight": [1, 3],
            "reg_lambda": [1.0, 3.0],
        }
    else:
        grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [300, 500],
            "subsample": [0.9],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "min_child_weight": [1],
            "reg_lambda": [1.0],
        }
    return list(ParameterGrid(grid))


def make_xgboost_exact(params: dict[str, Any], n_jobs: int, seed: int = RANDOM_STATE):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=len(LABELS),
        tree_method="exact",
        enable_categorical=False,
        eval_metric="mlogloss",
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=n_jobs,
        **params,
    )


def search_xgboost_exact_onehot(
    X_train,
    y_train,
    X_test,
    y_test,
    numeric_cols,
    cat_cols,
    n_jobs: int,
    cv_splits: int,
    wide: bool,
    local: bool,
):
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    param_grid = xgboost_exact_grid(wide=wide, local=local)
    records = []
    best_score = -np.inf
    best_record = None
    best_params = None

    for combo_idx, params in enumerate(param_grid, start=1):
        fold_macro_f1 = []
        fold_enrolled_f1 = []
        print(f"[XGB-EXACT] combo={combo_idx}/{len(param_grid)} params={params}", flush=True)
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train_encoded, y_train), start=1):
            X_tr = X_train_encoded[train_idx]
            X_va = X_train_encoded[valid_idx]
            y_tr_encoded = y_train_encoded[train_idx]
            y_va = y_train.iloc[valid_idx]

            model = make_xgboost_exact(params, n_jobs=n_jobs, seed=RANDOM_STATE + fold_idx)
            model.fit(X_tr, y_tr_encoded)
            pred_encoded = model.predict(X_va).astype(int)
            pred = label_encoder.inverse_transform(pred_encoded)
            macro_f1 = f1_score(y_va, pred, average="macro", zero_division=0)
            detail = class_metrics(y_va, pred)
            fold_macro_f1.append(macro_f1)
            fold_enrolled_f1.append(detail["enrolled_f1"])
            print(
                f"[XGB-EXACT-FOLD] combo={combo_idx}/{len(param_grid)} "
                f"fold={fold_idx}/{cv_splits} f1_macro={macro_f1:.4f} "
                f"enrolled_f1={detail['enrolled_f1']:.4f}",
                flush=True,
            )

        record = {
            **params,
            "cv_mean_f1_macro": float(np.mean(fold_macro_f1)),
            "cv_std_f1_macro": float(np.std(fold_macro_f1)),
            "cv_mean_enrolled_f1": float(np.mean(fold_enrolled_f1)),
            "cv_std_enrolled_f1": float(np.std(fold_enrolled_f1)),
            "tree_method": "exact",
            "categorical_handling": "onehot",
            "n_input_features": int(X_train_encoded.shape[1]),
        }
        records.append(record)
        save_metrics_table(records, XGB_EXACT_GRID)
        if record["cv_mean_f1_macro"] > best_score:
            best_score = record["cv_mean_f1_macro"]
            best_record = record
            best_params = params
            print(f"[XGB-EXACT-BEST] cv_mean_f1={best_score:.4f} params={params}", flush=True)

    assert best_params is not None and best_record is not None
    best_model = make_xgboost_exact(best_params, n_jobs=n_jobs)
    best_model.fit(X_train_encoded, y_train_encoded)
    pred_encoded = best_model.predict(X_test_encoded).astype(int)
    y_pred = label_encoder.inverse_transform(pred_encoded)
    y_prob = best_model.predict_proba(X_test_encoded)
    metrics = evaluate_text_predictions(
        "xgboost_exact_onehot_local_tuned" if local else "xgboost_exact_onehot_tuned",
        y_test,
        y_pred,
        y_prob,
        {
            "split_algorithm": "exact",
            "categorical_handling": "onehot",
            "max_bin": None,
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "n_input_features": int(X_train_encoded.shape[1]),
            "cv_best_f1_macro": best_record["cv_mean_f1_macro"],
            "cv_best_enrolled_f1": best_record["cv_mean_enrolled_f1"],
        },
    )
    XGB_EXACT_BEST.write_text(
        json.dumps(
            json_safe(
                {
                    "selection_metric": "cv_mean_f1_macro",
                    "cv_splits": cv_splits,
                    "n_jobs": n_jobs,
                    "wide": wide,
                    "local": local,
                    "best_params": best_params,
                    "best_cv_record": best_record,
                    "test_metrics": metrics,
                    "grid_path": str(XGB_EXACT_GRID),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics, best_params


def run_xgboost_hist_replay(X_train, y_train, X_test, y_test, cat_cols, n_jobs: int):
    X_train_native, X_test_native = prepare_native_categorical(X_train, X_test, cat_cols)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    hist_best = load_json(BOOST_METRICS_DIR / "hist_column_sampling_best_params.json")
    xgb_hist_params = hist_best["models"]["xgboost_hist"]["params"]
    params = {
        "objective": "multi:softprob",
        "num_class": len(LABELS),
        "tree_method": "hist",
        "enable_categorical": True,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 400,
        "subsample": 0.9,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "max_bin": int(xgb_hist_params["max_bin"]),
        "colsample_bytree": float(xgb_hist_params["column_fraction"]),
        "n_jobs": n_jobs,
    }
    model = XGBClassifier(**params)
    model.fit(X_train_native, y_train_encoded)
    y_pred_encoded = model.predict(X_test_native).astype(int)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_prob = model.predict_proba(X_test_native)

    return evaluate_text_predictions(
        "xgboost_hist_best_replay",
        y_test,
        y_pred,
        y_prob,
        {
            "split_algorithm": "hist",
            "categorical_handling": "native_category",
            "max_bin": params["max_bin"],
            "colsample_bytree": params["colsample_bytree"],
            "n_input_features": int(X_train_native.shape[1]),
        },
    ), params


def run_lightgbm_high_bin_variants(X_train, y_train, X_test, y_test, cat_cols, n_jobs: int, max_bins: list[int]):
    X_train_native, X_test_native = prepare_native_categorical(X_train, X_test, cat_cols)
    tuned_params = load_json(BOOST_METRICS_DIR / "lightgbm_tuned_best_params.json")
    hist_best = load_json(BOOST_METRICS_DIR / "hist_column_sampling_best_params.json")
    hist_column_fraction = float(hist_best["models"]["lightgbm"]["params"]["column_fraction"])

    variants = []
    variant_params = {}
    for column_policy, colsample in [
        ("tuned_colsample", float(tuned_params["colsample_bytree"])),
        ("hist_best_colsample", hist_column_fraction),
    ]:
        for max_bin in max_bins:
            params = {
                "objective": "multiclass",
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
                "verbosity": -1,
                "n_jobs": n_jobs,
                **tuned_params,
                "colsample_bytree": colsample,
                "max_bin": int(max_bin),
            }
            model = LGBMClassifier(**params)
            model.fit(X_train_native, y_train, categorical_feature=cat_cols)
            y_pred = model.predict(X_test_native)
            y_prob = model.predict_proba(X_test_native)
            name = f"lightgbm_high_bin_{column_policy}_maxbin_{max_bin}"
            variants.append(
                evaluate_text_predictions(
                    name,
                    y_test,
                    y_pred,
                    y_prob,
                    {
                        "split_algorithm": "hist_high_bin",
                        "categorical_handling": "native_category",
                        "max_bin": int(max_bin),
                        "colsample_bytree": colsample,
                        "n_input_features": int(X_train_native.shape[1]),
                    },
                )
            )
            variant_params[name] = params
    return variants, variant_params


def lightgbm_quantile_off_grid(wide: bool):
    if wide:
        grid = {
            "num_leaves": [31, 63, 127],
            "max_depth": [-1, 12],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [300, 500],
            "min_child_samples": [20, 40],
            "subsample": [0.8],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "max_bin": [255, 1023],
        }
    else:
        grid = {
            "num_leaves": [63],
            "max_depth": [-1],
            "learning_rate": [0.03, 0.05],
            "n_estimators": [300, 500],
            "min_child_samples": [20, 40],
            "subsample": [0.8],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "max_bin": [255, 1023],
        }
    return list(ParameterGrid(grid))


def make_lightgbm_quantile_off(params: dict[str, Any], n_jobs: int, seed: int = RANDOM_STATE):
    return LGBMClassifier(
        objective="multiclass",
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
        n_jobs=n_jobs,
        histogram_pool_size=-1,
        use_quantile_bins=False,
        **params,
    )


def search_lightgbm_quantile_off(X_train, y_train, X_test, y_test, cat_cols, n_jobs: int, cv_splits: int, wide: bool):
    X_train_native, X_test_native = prepare_native_categorical(X_train, X_test, cat_cols)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    param_grid = lightgbm_quantile_off_grid(wide=wide)
    records = []
    best_score = -np.inf
    best_record = None
    best_params = None

    for combo_idx, params in enumerate(param_grid, start=1):
        fold_macro_f1 = []
        fold_enrolled_f1 = []
        print(f"[LGB-QUANTILE-OFF] combo={combo_idx}/{len(param_grid)} params={params}", flush=True)
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_train_native, y_train), start=1):
            X_tr = X_train_native.iloc[train_idx].copy()
            X_va = X_train_native.iloc[valid_idx].copy()
            y_tr = y_train.iloc[train_idx]
            y_va = y_train.iloc[valid_idx]

            model = make_lightgbm_quantile_off(params, n_jobs=n_jobs, seed=RANDOM_STATE + fold_idx)
            model.fit(X_tr, y_tr, categorical_feature=cat_cols)
            pred = model.predict(X_va)
            macro_f1 = f1_score(y_va, pred, average="macro", zero_division=0)
            detail = class_metrics(y_va, pred)
            fold_macro_f1.append(macro_f1)
            fold_enrolled_f1.append(detail["enrolled_f1"])
            print(
                f"[LGB-QUANTILE-OFF-FOLD] combo={combo_idx}/{len(param_grid)} "
                f"fold={fold_idx}/{cv_splits} f1_macro={macro_f1:.4f} "
                f"enrolled_f1={detail['enrolled_f1']:.4f}",
                flush=True,
            )

        record = {
            **params,
            "cv_mean_f1_macro": float(np.mean(fold_macro_f1)),
            "cv_std_f1_macro": float(np.std(fold_macro_f1)),
            "cv_mean_enrolled_f1": float(np.mean(fold_enrolled_f1)),
            "cv_std_enrolled_f1": float(np.std(fold_enrolled_f1)),
            "histogram_pool_size": -1,
            "use_quantile_bins": False,
        }
        records.append(record)
        save_metrics_table(records, LGB_QUANTILE_OFF_GRID)
        if record["cv_mean_f1_macro"] > best_score:
            best_score = record["cv_mean_f1_macro"]
            best_record = record
            best_params = params
            print(f"[LGB-QUANTILE-OFF-BEST] cv_mean_f1={best_score:.4f} params={params}", flush=True)

    assert best_params is not None and best_record is not None
    best_model = make_lightgbm_quantile_off(best_params, n_jobs=n_jobs)
    best_model.fit(X_train_native, y_train, categorical_feature=cat_cols)
    y_pred = best_model.predict(X_test_native)
    y_prob = best_model.predict_proba(X_test_native)
    metrics = evaluate_text_predictions(
        "lightgbm_quantile_off_tuned",
        y_test,
        y_pred,
        y_prob,
        {
            "split_algorithm": "hist_quantile_off",
            "categorical_handling": "native_category",
            "max_bin": int(best_params["max_bin"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "n_input_features": int(X_train_native.shape[1]),
            "cv_best_f1_macro": best_record["cv_mean_f1_macro"],
            "cv_best_enrolled_f1": best_record["cv_mean_enrolled_f1"],
            "histogram_pool_size": -1,
            "use_quantile_bins": False,
        },
    )
    LGB_QUANTILE_OFF_BEST.write_text(
        json.dumps(
            json_safe(
                {
                    "selection_metric": "cv_mean_f1_macro",
                    "cv_splits": cv_splits,
                    "n_jobs": n_jobs,
                    "wide": wide,
                    "best_params": best_params,
                    "best_cv_record": best_record,
                    "test_metrics": metrics,
                    "grid_path": str(LGB_QUANTILE_OFF_GRID),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics, best_params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--lightgbm-max-bins", type=int, nargs="+", default=[255, 1023, 4095])
    parser.add_argument("--lightgbm-quantile-off-cv", type=int, default=3)
    parser.add_argument("--xgboost-exact-cv", type=int, default=3)
    parser.add_argument("--skip-lightgbm-quantile-off-search", action="store_true")
    parser.add_argument("--skip-xgboost-exact-search", action="store_true")
    parser.add_argument("--wide-lightgbm-quantile-off-search", action="store_true")
    parser.add_argument("--wide-xgboost-exact-search", action="store_true")
    parser.add_argument("--local-xgboost-exact-search", action="store_true")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, numeric_cols, cat_cols = load_data()
    BOOST_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    model_params = {}

    xgb_hist_metrics, xgb_hist_params = run_xgboost_hist_replay(X_train, y_train, X_test, y_test, cat_cols, args.n_jobs)
    records.append(xgb_hist_metrics)
    model_params[xgb_hist_metrics["model"]] = xgb_hist_params

    xgb_exact_metrics, xgb_exact_params = run_xgboost_exact_onehot(
        X_train, y_train, X_test, y_test, numeric_cols, cat_cols, args.n_jobs
    )
    records.append(xgb_exact_metrics)
    model_params[xgb_exact_metrics["model"]] = xgb_exact_params

    if not args.skip_xgboost_exact_search:
        xgb_tuned_metrics, xgb_tuned_params = search_xgboost_exact_onehot(
            X_train,
            y_train,
            X_test,
            y_test,
            numeric_cols,
            cat_cols,
            args.n_jobs,
            args.xgboost_exact_cv,
            args.wide_xgboost_exact_search,
            args.local_xgboost_exact_search,
        )
        records.append(xgb_tuned_metrics)
        model_params[xgb_tuned_metrics["model"]] = xgb_tuned_params

    lgb_records, lgb_params = run_lightgbm_high_bin_variants(
        X_train, y_train, X_test, y_test, cat_cols, args.n_jobs, args.lightgbm_max_bins
    )
    records.extend(lgb_records)
    model_params.update(lgb_params)

    if not args.skip_lightgbm_quantile_off_search:
        lgb_quantile_off_metrics, lgb_quantile_off_params = search_lightgbm_quantile_off(
            X_train,
            y_train,
            X_test,
            y_test,
            cat_cols,
            args.n_jobs,
            args.lightgbm_quantile_off_cv,
            args.wide_lightgbm_quantile_off_search,
        )
        records.append(lgb_quantile_off_metrics)
        model_params[lgb_quantile_off_metrics["model"]] = lgb_quantile_off_params

    df = pd.DataFrame(records).sort_values(["f1_macro", "accuracy"], ascending=[False, False])
    save_metrics_table(df.to_dict("records"), OUTPUT_METRICS)
    OUTPUT_METADATA.write_text(
        json.dumps(
            json_safe(
                {
                    "note": (
                        "XGBoost exact uses one-hot because exact split mode is not the native categorical path. "
                        "LightGBM has no exact split mode; high max_bin and quantile-off are still histogram checks."
                    ),
                    "n_jobs": args.n_jobs,
                    "xgboost_exact_cv": args.xgboost_exact_cv,
                    "wide_xgboost_exact_search": args.wide_xgboost_exact_search,
                    "local_xgboost_exact_search": args.local_xgboost_exact_search,
                    "xgboost_exact_grid_path": str(XGB_EXACT_GRID),
                    "xgboost_exact_best_path": str(XGB_EXACT_BEST),
                    "lightgbm_max_bins": args.lightgbm_max_bins,
                    "lightgbm_quantile_off_cv": args.lightgbm_quantile_off_cv,
                    "wide_lightgbm_quantile_off_search": args.wide_lightgbm_quantile_off_search,
                    "lightgbm_quantile_off_grid_path": str(LGB_QUANTILE_OFF_GRID),
                    "lightgbm_quantile_off_best_path": str(LGB_QUANTILE_OFF_BEST),
                    "categorical_columns": cat_cols,
                    "numeric_or_ordinal_columns": numeric_cols,
                    "params": model_params,
                    "metrics_path": str(OUTPUT_METRICS),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    display_cols = [
        "model",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "fmi",
        "roc_auc_ovr_macro",
        "dropout_accuracy",
        "enrolled_accuracy",
        "graduate_accuracy",
        "split_algorithm",
        "max_bin",
        "colsample_bytree",
    ]
    print(df[display_cols].to_string(index=False))
    print(f"saved metrics: {OUTPUT_METRICS}")
    print(f"saved metadata: {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
