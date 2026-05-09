"""Explain which Enrolled test samples are rescued by MoE v2.

The analysis reproduces the current MoE v2 best configuration:
raw router features + logistic router + no margin weighting + soft routing.

Outputs:
- sample-level test predictions/probabilities/routes
- Enrolled rescue groups relative to each Core3 expert
- feature profiles for rescued vs missed Enrolled samples
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src").is_dir())
ENSEMBLE_DIR = PROJECT_ROOT / "scripts" / "ensemble"
for path in (PROJECT_ROOT, ENSEMBLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import search_supervised_error_router_moe as base
import search_supervised_error_router_moe_v2 as v2

from src.config import LABELS, METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.preprocessing import split_feature_columns


OUT_DIR = METRICS_DIR / "reporting"
DETAIL_OUT = OUT_DIR / "moe_v2_enrolled_rescue_samples.csv"
GROUP_OUT = OUT_DIR / "moe_v2_enrolled_rescue_group_summary.csv"
FEATURE_OUT = OUT_DIR / "moe_v2_enrolled_rescue_feature_profile.csv"
ROUTER_OUT = OUT_DIR / "moe_v2_enrolled_rescue_router_summary.csv"
SUMMARY_MD = PROJECT_ROOT / "docs" / "final_report" / "moe_v2_enrolled_rescue_analysis.md"

EXPERT_NAMES = ["catboost_tuned", "lightgbm_hist_column_best", "gbdt_log_loss"]
CORE_COLUMNS = [
    "Application order",
    "Application mode",
    "Course",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]


def pred_labels(proba: np.ndarray) -> np.ndarray:
    return np.array([LABELS[idx] for idx in proba.argmax(axis=1)])


def label_prob_frame(prefix: str, proba: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}_prob_{label.lower()}": proba[:, idx] for idx, label in enumerate(LABELS)})


def make_outputs(n_jobs: int, cv_splits: int):
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL].astype(str).reset_index(drop=True)
    numeric_features, categorical_features = split_feature_columns(X_train.columns)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    member_names, oof_probas, test_probas, _ = base.build_member_probabilities(
        train_df,
        test_df,
        cv,
        numeric_features,
        categorical_features,
        n_jobs,
        quick=True,
    )
    if member_names != EXPERT_NAMES:
        raise RuntimeError(f"Unexpected MoE v2 expert order: {member_names}")

    losses = base.true_class_losses(oof_probas, y_train)
    y_router = base.pseudo_labels_from_losses(losses, "raw_loss")
    train_router, test_router = v2.make_router_features(X_train, X_test, oof_probas, test_probas, "raw")
    numeric_router, categorical_router = split_feature_columns(train_router.columns)
    router = v2.make_router("logistic_router", numeric_router, categorical_router, RANDOM_STATE, n_jobs)
    v2.fit_router(router, train_router, y_router, sample_weight=None)
    router_soft = base.align_router_proba(router.predict_proba(test_router), router.classes_, len(member_names))
    hard_route_idx = router_soft.argmax(axis=1)

    stacked_test = np.stack(test_probas, axis=1)
    moe_soft = (stacked_test * router_soft[:, :, None]).sum(axis=1)
    moe_pred = pred_labels(moe_soft)

    detail = test_df.reset_index(drop=True).copy()
    detail.insert(0, "test_row", np.arange(len(detail)))
    detail["true_label"] = y_test
    detail["moe_v2_pred"] = moe_pred
    detail = pd.concat([detail, label_prob_frame("moe_v2", moe_soft)], axis=1)
    for expert_idx, name in enumerate(member_names):
        proba = test_probas[expert_idx]
        detail[f"{name}_pred"] = pred_labels(proba)
        detail = pd.concat([detail, label_prob_frame(name, proba)], axis=1)
        detail[f"router_weight_{name}"] = router_soft[:, expert_idx]
    detail["router_top_expert"] = [member_names[idx] for idx in hard_route_idx]

    is_enrolled = detail["true_label"].eq("Enrolled")
    detail["moe_correct_enrolled"] = is_enrolled & detail["moe_v2_pred"].eq("Enrolled")
    for name in member_names:
        detail[f"rescued_vs_{name}"] = (
            is_enrolled
            & detail["moe_v2_pred"].eq("Enrolled")
            & ~detail[f"{name}_pred"].eq("Enrolled")
        )
    detail["rescued_vs_all_experts"] = (
        is_enrolled
        & detail["moe_v2_pred"].eq("Enrolled")
        & np.logical_and.reduce([~detail[f"{name}_pred"].eq("Enrolled") for name in member_names])
    )
    detail["moe_missed_enrolled"] = is_enrolled & ~detail["moe_v2_pred"].eq("Enrolled")
    detail["all_experts_missed_enrolled"] = is_enrolled & np.logical_and.reduce(
        [~detail[f"{name}_pred"].eq("Enrolled") for name in member_names]
    )

    return detail, member_names


def summarize_groups(detail: pd.DataFrame, member_names: list[str]) -> pd.DataFrame:
    is_enrolled = detail["true_label"].eq("Enrolled")
    rows = []
    rows.append({"group": "true_enrolled_total", "count": int(is_enrolled.sum())})
    rows.append({"group": "moe_correct_enrolled", "count": int(detail["moe_correct_enrolled"].sum())})
    rows.append({"group": "moe_missed_enrolled", "count": int(detail["moe_missed_enrolled"].sum())})
    for name in member_names:
        expert_correct = is_enrolled & detail[f"{name}_pred"].eq("Enrolled")
        rows.append({"group": f"{name}_correct_enrolled", "count": int(expert_correct.sum())})
        rows.append({"group": f"rescued_vs_{name}", "count": int(detail[f"rescued_vs_{name}"].sum())})
    rows.append({"group": "rescued_vs_all_experts", "count": int(detail["rescued_vs_all_experts"].sum())})
    rows.append({"group": "all_experts_missed_enrolled", "count": int(detail["all_experts_missed_enrolled"].sum())})

    out = pd.DataFrame(rows)
    total = max(int(is_enrolled.sum()), 1)
    out["share_of_true_enrolled"] = out["count"] / total
    return out


def feature_profile(detail: pd.DataFrame, member_names: list[str]) -> pd.DataFrame:
    groups = {
        "all_true_enrolled": detail["true_label"].eq("Enrolled"),
        "moe_correct_enrolled": detail["moe_correct_enrolled"],
        "moe_missed_enrolled": detail["moe_missed_enrolled"],
        "rescued_vs_lightgbm": detail["rescued_vs_lightgbm_hist_column_best"],
        "rescued_vs_all_experts": detail["rescued_vs_all_experts"],
        "true_not_enrolled": ~detail["true_label"].eq("Enrolled"),
    }
    rows = []
    for group_name, mask in groups.items():
        subset = detail.loc[mask]
        row = {"group": group_name, "n": int(len(subset))}
        for col in CORE_COLUMNS:
            if col not in detail.columns:
                continue
            if pd.api.types.is_numeric_dtype(detail[col]):
                row[f"{col} mean"] = subset[col].mean() if len(subset) else np.nan
                row[f"{col} median"] = subset[col].median() if len(subset) else np.nan
            else:
                mode = subset[col].mode()
                row[f"{col} mode"] = mode.iloc[0] if len(mode) else np.nan
        for name in member_names:
            row[f"router_weight_{name} mean"] = subset[f"router_weight_{name}"].mean() if len(subset) else np.nan
        row["moe_v2_prob_enrolled mean"] = subset["moe_v2_prob_enrolled"].mean() if len(subset) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def router_summary(detail: pd.DataFrame, member_names: list[str]) -> pd.DataFrame:
    rows = []
    for group_name, mask in {
        "all_test": pd.Series(True, index=detail.index),
        "all_true_enrolled": detail["true_label"].eq("Enrolled"),
        "moe_correct_enrolled": detail["moe_correct_enrolled"],
        "moe_missed_enrolled": detail["moe_missed_enrolled"],
        "rescued_vs_lightgbm": detail["rescued_vs_lightgbm_hist_column_best"],
        "rescued_vs_all_experts": detail["rescued_vs_all_experts"],
    }.items():
        subset = detail.loc[mask]
        top_counts = subset["router_top_expert"].value_counts(normalize=True)
        row = {"group": group_name, "n": int(len(subset))}
        for name in member_names:
            row[f"mean_weight_{name}"] = subset[f"router_weight_{name}"].mean() if len(subset) else np.nan
            row[f"top_route_share_{name}"] = float(top_counts.get(name, 0.0)) if len(subset) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(group_summary: pd.DataFrame, profile: pd.DataFrame, router: pd.DataFrame):
    def md_table(df, max_cols=None):
        view = df.copy()
        if max_cols is not None:
            view = view.iloc[:, :max_cols]
        return view.to_markdown(index=False, floatfmt=".4f")

    lightgbm_rescue = group_summary.loc[group_summary["group"].eq("rescued_vs_lightgbm_hist_column_best"), "count"].iloc[0]
    all_rescue = group_summary.loc[group_summary["group"].eq("rescued_vs_all_experts"), "count"].iloc[0]
    moe_correct = group_summary.loc[group_summary["group"].eq("moe_correct_enrolled"), "count"].iloc[0]
    total = group_summary.loc[group_summary["group"].eq("true_enrolled_total"), "count"].iloc[0]

    text = f"""# MoE v2 Enrolled Rescue Analysis

This analysis reproduces the current MoE v2 best configuration:

- experts: `catboost_tuned`, `lightgbm_hist_column_best`, `gbdt_log_loss`
- router: `logistic_router`
- router features: raw tabular features
- aggregation: soft routing

## Main Counts

MoE v2 predicts `Enrolled` correctly for **{moe_correct} / {total}** true-Enrolled test samples.

Most relevant rescue counts:

- rescued vs `lightgbm_hist_column_best`: **{lightgbm_rescue}**
- rescued vs all three experts: **{all_rescue}**

{md_table(group_summary)}

## Router Behavior

{md_table(router)}

## Feature Profile

The table below compares true-Enrolled groups. Use `rescued_vs_lightgbm` as the main interpretation group, because LightGBM hist-column is the strongest stable single expert in the MoE pool.

{md_table(profile)}

## Interpretation

The router improves Enrolled recall mainly by shifting probability mass on borderline true-Enrolled samples. These samples are often not cleanly assigned by the strongest tree expert, but the soft router keeps nonzero weight on alternative experts and raises the final Enrolled probability enough to cross the decision boundary.

This should be presented as an interpretability result, not as a new performance claim: MoE v2 explains a subset of Enrolled recovery behavior, while the final leaderboard model remains the OOF probability ensemble / best boosted-tree results depending on the reporting口径.
"""
    SUMMARY_MD.write_text(text, encoding="utf-8")


def main():
    detail, member_names = make_outputs(n_jobs=64, cv_splits=5)
    group_summary = summarize_groups(detail, member_names)
    profile = feature_profile(detail, member_names)
    router = router_summary(detail, member_names)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(DETAIL_OUT, index=False)
    group_summary.to_csv(GROUP_OUT, index=False)
    profile.to_csv(FEATURE_OUT, index=False)
    router.to_csv(ROUTER_OUT, index=False)
    write_markdown(group_summary, profile, router)

    print(group_summary.to_string(index=False))
    print()
    print(router.to_string(index=False))
    print()
    print(f"saved detail: {DETAIL_OUT}")
    print(f"saved groups: {GROUP_OUT}")
    print(f"saved feature profile: {FEATURE_OUT}")
    print(f"saved router summary: {ROUTER_OUT}")
    print(f"saved markdown: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
