"""Run t-SNE-friendly clustering baselines and save metrics.

Declaration:
- This script covers the course clustering task in plain Python.
- It compares at least two clustering methods on the processed dataset.

Tips:
- For clustering, avoid leaking the label into preprocessing.
- Interpret silhouette together with ARI/NMI, not in isolation.
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clustering import evaluate_clustering, get_clustering_models
from src.config import METRICS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET_COL
from src.evaluation import save_metrics_table
from src.preprocessing import build_preprocessor


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    X = train_df.drop(columns=[TARGET_COL])
    y = LabelEncoder().fit_transform(train_df[TARGET_COL])

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_transformed = preprocessor.fit_transform(X)

    records = []
    for model_name, model in get_clustering_models(n_clusters=3, random_state=RANDOM_STATE).items():
        cluster_labels = model.fit_predict(X_transformed)
        metrics = evaluate_clustering(X_transformed, y, cluster_labels)
        metrics["model"] = model_name
        records.append(metrics)

    save_metrics_table(records, METRICS_DIR / "clustering_metrics.csv")
    print(f"saved metrics: {METRICS_DIR / 'clustering_metrics.csv'}")


if __name__ == "__main__":
    main()
