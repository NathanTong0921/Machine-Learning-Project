from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "visualization" / "outputs"


def main():
    df = pd.read_csv(ROOT / "dataset" / "data.csv", sep=";")
    df.columns = df.columns.str.strip()

    zero_cols = [
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)",
    ]
    mask = df["Course"].eq(171) & df[zero_cols].eq(0).all(axis=1)
    sub = df.loc[mask].copy()
    sub["is_graduate"] = sub["Target"].eq("Graduate").astype(int)

    print("Course=171 + zero curricular total:", len(sub))
    print("Target distribution:")
    print(sub["Target"].value_counts().to_string())
    print("Non-Graduate count:", int((sub["Target"] != "Graduate").sum()))

    candidate_cols = [
        c
        for c in sub.columns
        if c not in zero_cols + ["Course", "Target", "is_graduate"]
    ]

    rows = []
    for col in candidate_cols:
        values = sub[col]
        if values.nunique(dropna=False) <= 1:
            continue

        if pd.api.types.is_numeric_dtype(values):
            grad = sub.loc[sub["is_graduate"].eq(1), col]
            nongrad = sub.loc[sub["is_graduate"].eq(0), col]
            pooled_std = sub[col].std(ddof=0)
            std_diff = np.nan if pooled_std == 0 else (grad.mean() - nongrad.mean()) / pooled_std
            mi = mutual_info_score(pd.qcut(values.rank(method="first"), q=min(5, len(values)), duplicates="drop"), sub["Target"])
            rows.append(
                {
                    "feature": col,
                    "type": "numeric",
                    "graduate_mean": grad.mean(),
                    "non_graduate_mean": nongrad.mean(),
                    "abs_std_diff": abs(std_diff),
                    "std_diff": std_diff,
                    "mutual_info_with_target": mi,
                    "graduate_top_value": np.nan,
                    "non_graduate_top_value": np.nan,
                    "graduate_top_ratio": np.nan,
                    "non_graduate_top_ratio": np.nan,
                }
            )
        else:
            grad_vc = sub.loc[sub["is_graduate"].eq(1), col].value_counts(normalize=True, dropna=False)
            nongrad_vc = sub.loc[sub["is_graduate"].eq(0), col].value_counts(normalize=True, dropna=False)
            mi = mutual_info_score(values.astype(str), sub["Target"])
            rows.append(
                {
                    "feature": col,
                    "type": "categorical",
                    "graduate_mean": np.nan,
                    "non_graduate_mean": np.nan,
                    "abs_std_diff": np.nan,
                    "std_diff": np.nan,
                    "mutual_info_with_target": mi,
                    "graduate_top_value": grad_vc.index[0],
                    "non_graduate_top_value": nongrad_vc.index[0],
                    "graduate_top_ratio": grad_vc.iloc[0],
                    "non_graduate_top_ratio": nongrad_vc.iloc[0],
                }
            )

    feature_rank = pd.DataFrame(rows).sort_values(
        ["mutual_info_with_target", "abs_std_diff"], ascending=False
    )
    feature_rank.to_csv(OUT / "course171_zero_curricular_feature_separation.csv", index=False)
    print("\nTop separating features within Course=171 + zero curricular:")
    print(feature_rank.head(20).to_string(index=False))

    # Simple human-readable rules for predicting Graduate inside this subgroup.
    rule_defs = []
    for adm in [120, 125, 130, 135, 140, 145]:
        for prev in [120, 125, 130, 135, 140, 145]:
            rule_defs.append((f"Admission>={adm} & PreviousGrade>={prev}", (sub["Admission grade"] >= adm) & (sub["Previous qualification (grade)"] >= prev)))
    rule_defs.extend(
        [
            ("Application mode == 1", sub["Application mode"].eq(1)),
            ("Application mode == 1 & Admission>=130 & PreviousGrade>=130", sub["Application mode"].eq(1) & (sub["Admission grade"] >= 130) & (sub["Previous qualification (grade)"] >= 130)),
            ("Displaced == 1 & Admission>=130 & PreviousGrade>=130", sub["Displaced"].eq(1) & (sub["Admission grade"] >= 130) & (sub["Previous qualification (grade)"] >= 130)),
            ("Tuition up-to-date == 1 & Debtor == 0", sub["Tuition fees up to date"].eq(1) & sub["Debtor"].eq(0)),
            ("Gender == 1 & Admission>=130 & PreviousGrade>=130", sub["Gender"].eq(1) & (sub["Admission grade"] >= 130) & (sub["Previous qualification (grade)"] >= 130)),
        ]
    )

    rule_rows = []
    for name, hit in rule_defs:
        hit = hit.fillna(False)
        n_hit = int(hit.sum())
        if n_hit == 0:
            continue
        tp = int((hit & sub["is_graduate"].eq(1)).sum())
        fp = int((hit & sub["is_graduate"].eq(0)).sum())
        fn = int((~hit & sub["is_graduate"].eq(1)).sum())
        precision = tp / n_hit
        recall = tp / int(sub["is_graduate"].sum())
        rule_rows.append(
            {
                "rule": name,
                "hits": n_hit,
                "graduate_hits": tp,
                "non_graduate_hits": fp,
                "precision_for_graduate": precision,
                "recall_of_75_graduates": recall,
                "false_positive_rate_among_105_non_graduates": fp / int(sub["is_graduate"].eq(0).sum()),
            }
        )

    rule_eval = pd.DataFrame(rule_rows).sort_values(
        ["precision_for_graduate", "recall_of_75_graduates"], ascending=False
    )
    rule_eval.to_csv(OUT / "course171_zero_curricular_rule_candidates.csv", index=False)
    print("\nTop candidate rules:")
    print(rule_eval.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
