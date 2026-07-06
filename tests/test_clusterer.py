import inspect

import numpy as np

from src.clusterer import DEFAULT_K_MAX, DEFAULT_K_MIN, cluster_documents, choose_best_k


def test_choose_best_k_defaults_match_the_shared_constants():
    # Guards against the two functions in this module silently drifting
    # apart (they once had different k_min/k_max defaults from each other).
    params = inspect.signature(choose_best_k).parameters
    assert params["k_min"].default == DEFAULT_K_MIN
    assert params["k_max"].default == DEFAULT_K_MAX


def test_cluster_documents_defaults_match_the_shared_constants():
    params = inspect.signature(cluster_documents).parameters
    assert params["k_min"].default == DEFAULT_K_MIN
    assert params["k_max"].default == DEFAULT_K_MAX


def _two_separated_blobs():
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=[10, 0], scale=0.05, size=(15, 2))
    blob_b = rng.normal(loc=[0, 10], scale=0.05, size=(15, 2))
    return np.vstack([blob_a, blob_b])


def test_choose_best_k_picks_two_for_two_well_separated_blobs():
    vectors = _two_separated_blobs()
    k = choose_best_k(vectors, k_min=2, k_max=6)
    assert k == 2


def test_cluster_documents_labels_length_matches_input():
    vectors = _two_separated_blobs()
    labels = cluster_documents(vectors, k_min=2, k_max=6)
    assert len(labels) == len(vectors)


def test_cluster_documents_separates_the_two_blobs():
    vectors = _two_separated_blobs()
    labels = cluster_documents(vectors, k_min=2, k_max=6)
    first_half_labels = set(labels[:15])
    second_half_labels = set(labels[15:])
    assert len(first_half_labels) == 1
    assert len(second_half_labels) == 1
    assert first_half_labels != second_half_labels


def test_single_document_returns_single_cluster():
    vectors = np.array([[1.0, 2.0, 3.0]])
    labels = cluster_documents(vectors, k_min=2, k_max=10)
    assert list(labels) == [0]


def test_too_few_documents_for_k_min_returns_single_cluster():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = cluster_documents(vectors, k_min=3, k_max=10)
    assert list(labels) == [0, 0]


def test_reproducible_with_fixed_random_state():
    vectors = _two_separated_blobs()
    labels_a = cluster_documents(vectors, k_min=2, k_max=6, random_state=42)
    labels_b = cluster_documents(vectors, k_min=2, k_max=6, random_state=42)
    assert list(labels_a) == list(labels_b)


def test_fixed_k_overrides_automatic_silhouette_selection():
    # この2つの塊に対してはシルエットスコアはk=2を選ぶはずだが
    # （上のテスト参照）、明示的なfixed_kは必ず優先されなければならない。
    vectors = _two_separated_blobs()
    labels = cluster_documents(vectors, fixed_k=5, random_state=42)
    assert len(set(labels)) == 5


def test_fixed_k_of_one_returns_single_cluster():
    vectors = _two_separated_blobs()
    labels = cluster_documents(vectors, fixed_k=1, random_state=42)
    assert list(labels) == [0] * len(vectors)


def test_fixed_k_larger_than_sample_count_is_clamped():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = cluster_documents(vectors, fixed_k=10, random_state=42)
    assert len(set(labels)) <= len(vectors)
