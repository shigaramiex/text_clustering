import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def choose_best_k(
    vectors: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> int:
    """Pick the k in [k_min, k_max] with the highest silhouette score."""
    n_samples = len(vectors)
    effective_k_max = min(k_max, n_samples - 1)
    if effective_k_max < k_min:
        return 1

    best_k = k_min
    best_score = -1.0
    for k in range(k_min, effective_k_max + 1):
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(
            vectors
        )
        score = silhouette_score(vectors, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_documents(
    vectors: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> np.ndarray:
    """Cluster document vectors, automatically choosing k via silhouette score."""
    n_samples = len(vectors)
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)

    k = choose_best_k(vectors, k_min=k_min, k_max=k_max, random_state=random_state)
    if k <= 1:
        return np.zeros(n_samples, dtype=int)

    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(vectors)
