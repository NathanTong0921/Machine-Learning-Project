#!/usr/bin/env python3
"""Compile cross-seed mean/std summaries from seed-level metric files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "metrics" / "reporting" / "all_model_results_cross_seed_summary.csv"

METRIC_MAP = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision macro",
    "recall_macro": "Recall macro",
    "f1_macro": "F1 macro",
    "f1_weighted": "F1 weighted",
    "roc_auc_ovr_macro": "AUC",
    "fmi": "FMI",
    "dropout_precision": "Dropout precision",
    "dropout_recall": "Dropout recall",
    "dropout_f1": "Dropout F1",
    "dropout_accuracy": "Dropout acc",
    "enrolled_precision": "Enrolled precision",
    "enrolled_recall": "Enrolled recall",
    "enrolled_f1": "Enrolled F1",
    "enrolled_accuracy": "Enrolled acc",
    "graduate_precision": "Graduate precision",
    "graduate_recall": "Graduate recall",
    "graduate_f1": "Graduate F1",
    "graduate_accuracy": "Graduate acc",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    aliases = {
        "dropout_acc": "dropout_accuracy",
        "enrolled_acc": "enrolled_accuracy",
        "graduate_acc": "graduate_accuracy",
    }
    return df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})


def read_classic_boosting() -> pd.DataFrame:
    rel = Path("models/metrics/boost/classic_boosting_cross_seed_metrics.csv")
    df = normalize_columns(pd.read_csv(ROOT / rel))
    df["Family"] = "boost"
    df["Set"] = "Testing"
    df["Source"] = str(rel)
    return df


def read_transformer_seed_file(rel: str, model_name: str) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(ROOT / rel))
    df["model"] = model_name
    df["Family"] = "transformer"
    df["Set"] = "Testing"
    df["Source"] = rel
    return df


def read_cross_seed_ensemble() -> pd.DataFrame:
    rel = Path("models/metrics/ensemble/cross_seed_stacking_metrics.csv")
    raw = normalize_columns(pd.read_csv(ROOT / rel))
    rows = []
    pattern = re.compile(r"^(?P<base>.+)_seed(?P<seed>\d+)_(?P<set>oof|test)$")
    for _, row in raw.iterrows():
        match = pattern.match(str(row.get("model", "")))
        if not match:
            continue
        record = row.to_dict()
        record["model"] = match.group("base")
        record["seed"] = int(match.group("seed"))
        record["Set"] = "OOF" if match.group("set") == "oof" else "Testing"
        record["Family"] = "ensemble"
        record["Source"] = str(rel)
        rows.append(record)
    return pd.DataFrame(rows)


def collect_seed_rows() -> pd.DataFrame:
    frames = [
        read_classic_boosting(),
        read_transformer_seed_file(
            "models/metrics/transformer/ft_seed_stability.csv",
            "ft_transformer_seed_stability",
        ),
        read_transformer_seed_file(
            "models/metrics/transformer/viewwise_gated_ft_seed_stability.csv",
            "viewwise_gated_ft_seed_stability",
        ),
        read_cross_seed_ensemble(),
    ]
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "seed" not in df.columns:
        raise RuntimeError("No seed column found after collecting seed-level files.")
    df = df.dropna(subset=["seed"])
    df["seed"] = df["seed"].astype(int)
    return df


def summarize(group: pd.DataFrame) -> pd.Series:
    out = {
        "Model": group["model"].iloc[0],
        "Family": group["Family"].iloc[0],
        "Set": group["Set"].iloc[0],
        "n_seeds": int(group["seed"].nunique()),
        "seeds": ",".join(str(x) for x in sorted(group["seed"].unique())),
        "best_seed_by_f1_macro": "",
        "best_f1_macro": pd.NA,
        "worst_seed_by_f1_macro": "",
        "worst_f1_macro": pd.NA,
        "Source": " | ".join(sorted(group["Source"].astype(str).unique())),
    }
    if "f1_macro" in group.columns and group["f1_macro"].notna().any():
        best_idx = group["f1_macro"].idxmax()
        worst_idx = group["f1_macro"].idxmin()
        out["best_seed_by_f1_macro"] = int(group.loc[best_idx, "seed"])
        out["best_f1_macro"] = group.loc[best_idx, "f1_macro"]
        out["worst_seed_by_f1_macro"] = int(group.loc[worst_idx, "seed"])
        out["worst_f1_macro"] = group.loc[worst_idx, "f1_macro"]

    for col, label in METRIC_MAP.items():
        if col not in group.columns:
            continue
        values = pd.to_numeric(group[col], errors="coerce").dropna()
        if values.empty:
            continue
        out[f"{label} mean"] = values.mean()
        out[f"{label} std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        out[f"{label} min"] = values.min()
        out[f"{label} max"] = values.max()
    return pd.Series(out)


def main() -> None:
    seed_rows = collect_seed_rows()
    records = []
    for _, group in seed_rows.groupby(["Family", "Set", "model"], dropna=False):
        records.append(summarize(group).to_dict())
    summary = pd.DataFrame(records)
    if "F1 macro mean" in summary.columns:
        summary = summary.sort_values(["Set", "F1 macro mean"], ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(ROOT)}")
    print(f"Rows: {len(summary)}")


if __name__ == "__main__":
    main()
