from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baseline.search_svm_hyperparameters import make_dense_preprocessor  # noqa: E402
from src.config import CATEGORICAL_COLUMNS, PROCESSED_DIR, RANDOM_STATE, TARGET_COL  # noqa: E402
from src.preprocessing import split_feature_columns  # noqa: E402


OUT_DIR = PROJECT_ROOT / "visualization" / "noise_feature_analysis"
LABELS = ["Dropout", "Enrolled", "Graduate"]
BEST_DENSE_RBF = {
    "kernel": "rbf",
    "C": 30.0,
    "gamma": 0.003,
    "class_weight": {"Dropout": 1.0, "Enrolled": 1.25, "Graduate": 1.0},
}


DROP_SETS = {
    "none_baseline": [],
    "drop_macro": ["Unemployment rate", "Inflation rate", "GDP"],
    "drop_credited_without_eval": [
        "Curricular units 1st sem (credited)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (without evaluations)",
    ],
    "drop_rare_binary": ["International", "Educational special needs"],
    "drop_macro_credited_without_eval_rare_binary": [
        "Unemployment rate",
        "Inflation rate",
        "GDP",
        "Curricular units 1st sem (credited)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (without evaluations)",
        "International",
        "Educational special needs",
    ],
    "drop_application_order": ["Application order"],
    "drop_nacionality": ["Nacionality"],
    "drop_parent_qual_occ": [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ],
}


def build_pipeline(columns):
    numeric_features, categorical_features = split_feature_columns(columns)
    preprocessor = make_dense_preprocessor(numeric_features, categorical_features)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", SVC(**BEST_DENSE_RBF, probability=False, cache_size=2048, random_state=RANDOM_STATE)),
        ]
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    y = train[TARGET_COL]
    X = train.drop(columns=[TARGET_COL])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    records = []
    for name, drop_cols in DROP_SETS.items():
        keep_cols = [c for c in X.columns if c not in set(drop_cols)]
        pipe = build_pipeline(keep_cols)
        pred = cross_val_predict(pipe, X[keep_cols], y, cv=cv, n_jobs=1)
        records.append(
            {
                "variant": name,
                "dropped_features": "|".join(drop_cols),
                "n_dropped": len(drop_cols),
                "n_features_raw": len(keep_cols),
                "cv_macro_f1": f1_score(y, pred, average="macro", labels=LABELS),
                "cv_accuracy": accuracy_score(y, pred),
                "cv_enrolled_f1": f1_score(y, pred, labels=["Enrolled"], average="macro", zero_division=0),
                "cv_enrolled_recall": recall_score(y, pred, labels=["Enrolled"], average="macro", zero_division=0),
            }
        )

    out = pd.DataFrame(records)
    base = out.loc[out["variant"] == "none_baseline", "cv_macro_f1"].iloc[0]
    out["delta_macro_f1_vs_baseline"] = out["cv_macro_f1"] - base
    out = out.sort_values("cv_macro_f1", ascending=False)
    out.to_csv(OUT_DIR / "svm_drop_probe_cv.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
