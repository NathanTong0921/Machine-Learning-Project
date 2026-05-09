from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CATEGORICAL_COLUMNS, FEATURE_GROUPS, TARGET_COL  # noqa: E402


OUT_DIR = PROJECT_ROOT / "visualization" / "noise_feature_analysis"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
LABEL_ORDER = ["Dropout", "Enrolled", "Graduate"]
RARE_THRESHOLD = 0.01


def ensure_target_labels(y):
    y = y.astype(str).str.strip()
    if set(y.unique()) <= {"0", "1", "2"}:
        mapping = {"0": "Dropout", "1": "Enrolled", "2": "Graduate"}
        y = y.map(mapping)
    return y


def feature_group(feature):
    for group, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return group
    return "other"


def cramers_v(x, y):
    table = pd.crosstab(x.astype(str), y.astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    phi2 = chi2 / max(n, 1)
    r, k = table.shape
    denom = max(min(k - 1, r - 1), 1)
    return float(np.sqrt(phi2 / denom))


def eta_squared_numeric(x, y):
    valid = x.notna() & y.notna()
    x = x[valid].astype(float)
    y = y[valid]
    if x.nunique() <= 1:
        return 0.0
    grand = x.mean()
    ss_between = 0.0
    for _, vals in x.groupby(y):
        ss_between += len(vals) * (vals.mean() - grand) ** 2
    ss_total = ((x - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def total_variation(train_s, test_s):
    train_freq = train_s.astype(str).value_counts(normalize=True, dropna=False)
    test_freq = test_s.astype(str).value_counts(normalize=True, dropna=False)
    idx = train_freq.index.union(test_freq.index)
    diff = (train_freq.reindex(idx, fill_value=0) - test_freq.reindex(idx, fill_value=0)).abs()
    return float(0.5 * diff.sum()), float(diff.max())


def rare_category_stats(train_s, test_s):
    train_freq = train_s.astype(str).value_counts(normalize=True, dropna=False)
    rare_levels = train_freq[train_freq < RARE_THRESHOLD]
    train_levels = set(train_freq.index)
    test_freq = test_s.astype(str).value_counts(normalize=True, dropna=False)
    unseen = [lvl for lvl in test_freq.index if lvl not in train_levels]
    singleton_count = int((train_s.astype(str).value_counts(dropna=False) == 1).sum())
    return {
        "rare_threshold": RARE_THRESHOLD,
        "rare_level_count": int(rare_levels.shape[0]),
        "rare_train_mass": float(rare_levels.sum()),
        "singleton_level_count": singleton_count,
        "unseen_test_level_count": int(len(unseen)),
        "unseen_test_mass": float(test_freq.loc[unseen].sum()) if unseen else 0.0,
    }


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def univariate_cv_scores(df, y, features, categorical):
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for feature in features:
        X = df[[feature]]
        if feature in categorical:
            preprocess = ColumnTransformer(
                [("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]), [feature])]
            )
        else:
            preprocess = ColumnTransformer(
                [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), [feature])]
            )
        clf = Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        try:
            pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
            rows.append(
                {
                    "feature": feature,
                    "univariate_macro_f1": f1_score(y, pred, average="macro", labels=LABEL_ORDER),
                    "univariate_enrolled_recall": recall_score(
                        y, pred, labels=["Enrolled"], average="macro", zero_division=0
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "feature": feature,
                    "univariate_macro_f1": np.nan,
                    "univariate_enrolled_recall": np.nan,
                    "univariate_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def compute_mi(df, y, features, categorical):
    X = df[features].copy()
    discrete = []
    for feature in features:
        if feature in categorical:
            X[feature] = X[feature].astype(str).fillna("__MISSING__").astype("category").cat.codes
            discrete.append(True)
        else:
            X[feature] = X[feature].astype(float).fillna(X[feature].median())
            discrete.append(False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=42)
    return pd.DataFrame({"feature": features, "mutual_info": mi})


def normalized_rank_score(s, ascending=False):
    ranks = s.rank(method="average", ascending=ascending, na_option="bottom")
    if len(s) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (ranks - 1) / (len(s) - 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = ensure_target_labels(train[TARGET_COL])
    features = [c for c in train.columns if c != TARGET_COL]
    categorical = sorted(set(CATEGORICAL_COLUMNS).intersection(features))
    numeric = [c for c in features if c not in categorical]

    summary_rows = []
    shift_rows = []
    rare_rows = []
    assoc_rows = []

    for feature in features:
        tr = train[feature]
        te = test[feature]
        is_cat = feature in categorical
        row = {
            "feature": feature,
            "group": feature_group(feature),
            "kind": "categorical" if is_cat else "numeric",
            "train_missing_rate": float(tr.isna().mean()),
            "test_missing_rate": float(te.isna().mean()),
            "train_unique": int(tr.nunique(dropna=False)),
            "test_unique": int(te.nunique(dropna=False)),
            "train_top_freq": float(tr.astype(str).value_counts(normalize=True, dropna=False).iloc[0]),
        }
        summary_rows.append(row)

        if is_cat:
            tv, max_delta = total_variation(tr, te)
            rare = rare_category_stats(tr, te)
            rare_rows.append({"feature": feature, **rare})
            shift_rows.append(
                {
                    "feature": feature,
                    "kind": "categorical",
                    "shift_metric": tv,
                    "shift_name": "total_variation",
                    "max_frequency_delta": max_delta,
                    "unseen_test_mass": rare["unseen_test_mass"],
                    "unseen_test_level_count": rare["unseen_test_level_count"],
                }
            )
            assoc_rows.append(
                {
                    "feature": feature,
                    "association_metric": cramers_v(tr, y),
                    "association_name": "cramers_v",
                }
            )
        else:
            ks = ks_2samp(tr.dropna(), te.dropna())
            tr_std = tr.std(ddof=0)
            mean_std_diff = abs(tr.mean() - te.mean()) / tr_std if tr_std > 0 else 0.0
            shift_rows.append(
                {
                    "feature": feature,
                    "kind": "numeric",
                    "shift_metric": float(ks.statistic),
                    "shift_name": "ks_statistic",
                    "ks_pvalue": float(ks.pvalue),
                    "mean_std_diff": float(mean_std_diff),
                }
            )
            assoc_rows.append(
                {
                    "feature": feature,
                    "association_metric": eta_squared_numeric(tr, y),
                    "association_name": "eta_squared",
                }
            )

    summary = pd.DataFrame(summary_rows)
    shift = pd.DataFrame(shift_rows)
    rare_df = pd.DataFrame(rare_rows).sort_values(
        ["rare_train_mass", "rare_level_count", "unseen_test_mass"], ascending=False
    )
    assoc = pd.DataFrame(assoc_rows)
    mi = compute_mi(train, y, features, categorical)
    cv_scores = univariate_cv_scores(train, y, features, categorical)

    diag = (
        summary.merge(shift, on=["feature", "kind"], how="left")
        .merge(assoc, on="feature", how="left")
        .merge(mi, on="feature", how="left")
        .merge(cv_scores, on="feature", how="left")
        .merge(rare_df, on="feature", how="left")
    )
    if "unseen_test_mass_x" in diag:
        diag["unseen_test_mass"] = diag["unseen_test_mass_x"].fillna(diag.get("unseen_test_mass_y", 0))
    if "unseen_test_level_count_x" in diag:
        diag["unseen_test_level_count"] = diag["unseen_test_level_count_x"].fillna(
            diag.get("unseen_test_level_count_y", 0)
        )
    diag = diag.drop(
        columns=[
            col
            for col in [
                "unseen_test_mass_x",
                "unseen_test_mass_y",
                "unseen_test_level_count_x",
                "unseen_test_level_count_y",
            ]
            if col in diag.columns
        ]
    )
    for col in [
        "rare_level_count",
        "rare_train_mass",
        "singleton_level_count",
        "unseen_test_mass",
        "unseen_test_level_count",
    ]:
        if col not in diag:
            diag[col] = 0
        diag[col] = diag[col].fillna(0)

    diag["low_signal_score"] = (
        0.40 * normalized_rank_score(diag["mutual_info"], ascending=False)
        + 0.35 * normalized_rank_score(diag["association_metric"], ascending=False)
        + 0.25 * normalized_rank_score(diag["univariate_macro_f1"], ascending=False)
    )
    diag["svm_noise_pressure"] = (
        0.30 * normalized_rank_score(diag["shift_metric"], ascending=False)
        + 0.25 * normalized_rank_score(diag["rare_train_mass"], ascending=False)
        + 0.15 * normalized_rank_score(diag["unseen_test_mass"], ascending=False)
        + 0.15 * normalized_rank_score(diag["train_unique"], ascending=False)
        + 0.15 * normalized_rank_score(diag["train_top_freq"], ascending=False)
    )
    diag["harmful_for_svm_score"] = 0.62 * diag["low_signal_score"] + 0.38 * diag["svm_noise_pressure"]
    diag = diag.sort_values("harmful_for_svm_score", ascending=False)

    summary.sort_values(["kind", "feature"]).to_csv(OUT_DIR / "feature_basic_diagnostics.csv", index=False)
    rare_df.to_csv(OUT_DIR / "category_rare_rate.csv", index=False)
    shift.sort_values("shift_metric", ascending=False).to_csv(OUT_DIR / "train_test_shift.csv", index=False)
    assoc.sort_values("association_metric").to_csv(OUT_DIR / "feature_target_association.csv", index=False)
    mi.sort_values("mutual_info").to_csv(OUT_DIR / "feature_mutual_info.csv", index=False)
    cv_scores.sort_values("univariate_macro_f1").to_csv(OUT_DIR / "univariate_cv_scores.csv", index=False)
    diag.to_csv(OUT_DIR / "feature_noise_ranking.csv", index=False)

    top = diag.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#b94b4b" if k == "categorical" else "#4778a8" for k in top["kind"]]
    ax.barh(top["feature"], top["harmful_for_svm_score"], color=colors)
    ax.set_xlabel("Heuristic SVM harmful/noise score")
    ax.set_title("Top noise-like / low-value feature candidates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_noise_feature_candidates.png", dpi=180)
    plt.close(fig)

    shift_top = shift.sort_values("shift_metric", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(shift_top["feature"], shift_top["shift_metric"], color="#7f6aa5")
    ax.set_xlabel("Shift metric: KS for numeric, TV for categorical")
    ax.set_title("Largest train/test distribution shifts")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_train_test_shift.png", dpi=180)
    plt.close(fig)

    cat_plot = rare_df.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(cat_plot["feature"], cat_plot["rare_train_mass"], color="#b8863b")
    ax.set_xlabel(f"Train mass in categories below {RARE_THRESHOLD:.0%}")
    ax.set_title("Rare-category mass by categorical feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rare_category_mass.png", dpi=180)
    plt.close(fig)

    focus_cols = [
        "feature",
        "kind",
        "group",
        "harmful_for_svm_score",
        "low_signal_score",
        "svm_noise_pressure",
        "mutual_info",
        "association_metric",
        "univariate_macro_f1",
        "univariate_enrolled_recall",
        "shift_metric",
        "train_unique",
        "rare_train_mass",
        "unseen_test_mass",
        "train_top_freq",
    ]
    report = [
        "# Noise Feature Diagnostics",
        "",
        f"Train rows: {len(train)}, test rows: {len(test)}, features: {len(features)}.",
        "",
        "## Top heuristic SVM noise candidates",
        "",
        diag[focus_cols].head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Largest train/test shifts",
        "",
        shift.sort_values("shift_metric", ascending=False).head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Categorical rare-category pressure",
        "",
        rare_df.head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Lowest univariate CV signal",
        "",
        cv_scores.sort_values("univariate_macro_f1").head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (OUT_DIR / "noise_feature_diagnostics_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Wrote diagnostics to {OUT_DIR}")


if __name__ == "__main__":
    main()
