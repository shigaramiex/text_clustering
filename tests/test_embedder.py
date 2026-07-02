import numpy as np

from src.embedder import embed_texts


def test_output_shape_matches_input_count_and_model_dim():
    vectors = embed_texts(["猫 犬 動物", "サッカー 野球 スポーツ", "猫 犬 動物"])
    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0


def test_vectors_are_l2_normalized():
    vectors = embed_texts(["猫 犬 動物", "サッカー 野球 スポーツ"])
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_similar_texts_are_closer_than_dissimilar_texts():
    vectors = embed_texts(
        [
            "猫 犬 動物 ペット",
            "犬 猫 ペット 動物",
            "サッカー 野球 スポーツ 試合",
        ]
    )
    sim_same_topic = vectors[0] @ vectors[1]
    sim_diff_topic = vectors[0] @ vectors[2]
    assert sim_same_topic > sim_diff_topic


def test_empty_list_returns_empty_array():
    vectors = embed_texts([])
    assert vectors.shape[0] == 0
