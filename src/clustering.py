"""Clustering baselines and evaluation helpers."""

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def get_clustering_models(n_clusters=3, random_state=42):
    return {
        "kmeans": KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
    }


def evaluate_clustering(X, y_true, cluster_labels):
    results = {}
    if len(set(cluster_labels)) > 1:
        results["silhouette"] = silhouette_score(X, cluster_labels)
    else:
        results["silhouette"] = float("nan")

    if y_true is not None:
        results["ari"] = adjusted_rand_score(y_true, cluster_labels)
        results["nmi"] = normalized_mutual_info_score(y_true, cluster_labels)
    return results
