import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# クラスタ数関連の唯一の情報源（single source of truth）。
# pipeline.py・gui.pyはここから値を参照し、各自でハードコードしない。
DEFAULT_K_MIN = 20
DEFAULT_K_MAX = 30
DEFAULT_FIXED_K = 20


def choose_best_k(
    vectors: np.ndarray,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    random_state: int = 42,
) -> int:
    """[k_min, k_max]の範囲でシルエットスコアが最も高いkを選ぶ。"""
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
    vectors: np.ndarray,
    k_min: int = DEFAULT_K_MIN,
    k_max: int = DEFAULT_K_MAX,
    random_state: int = 42,
    fixed_k: int | None = None,
) -> np.ndarray:
    """文書ベクトルをクラスタリングする。

    fixed_kが指定された場合は、そのクラスタ数（サンプル数でクランプ）を
    自動選択せずそのまま使用する。指定がなければ[k_min, k_max]の範囲で
    シルエットスコアによりkを自動的に選ぶ。
    """
    n_samples = len(vectors)
    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)

    if fixed_k is not None:
        k = min(fixed_k, n_samples)
    else:
        k = choose_best_k(vectors, k_min=k_min, k_max=k_max, random_state=random_state)

    if k <= 1:
        return np.zeros(n_samples, dtype=int)

    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(vectors)
