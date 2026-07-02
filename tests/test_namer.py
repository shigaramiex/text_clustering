import numpy as np

from src.namer import assign_cluster_names, sanitize_folder_name


def test_representative_noun_matches_dominant_topic():
    token_lists = [
        ["猫", "犬", "動物", "散歩"],
        ["犬", "猫", "動物", "餌"],
        ["猫", "動物", "毛"],
        ["サッカー", "野球", "試合", "選手"],
        ["野球", "試合", "監督"],
        ["サッカー", "試合", "得点"],
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])

    names = assign_cluster_names(token_lists, labels)

    assert names[0] in {"猫", "犬", "動物"}
    assert names[1] in {"サッカー", "野球", "試合"}


def test_returns_a_name_for_every_unique_label():
    token_lists = [["猫", "犬"], ["サッカー", "野球"], ["旅行", "電車"]]
    labels = np.array([0, 1, 2])

    names = assign_cluster_names(token_lists, labels)

    assert set(names.keys()) == {0, 1, 2}


def test_ubiquitous_term_is_not_chosen_over_distinguishing_terms():
    # "映画"(movie) appears in every document of both clusters, so it
    # carries no information distinguishing the two groups. The word that
    # is prominent in one cluster but absent from the other ("アニメ" /
    # "ハリウッド") should be picked instead, so the two cluster names are
    # not both "映画" (a real report: two movie clusters both named
    # "映画"/"映画_2" gave no sense of how they differ).
    token_lists = [
        ["映画", "映画", "アニメ", "監督"],
        ["映画", "映画", "アニメ", "声優"],
        ["映画", "映画", "アニメ", "作画"],
        ["映画", "映画", "ハリウッド", "俳優"],
        ["映画", "映画", "ハリウッド", "主演"],
        ["映画", "映画", "ハリウッド", "撮影"],
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])

    names = assign_cluster_names(token_lists, labels)

    assert names[0] == "アニメ"
    assert names[1] == "ハリウッド"
    assert "映画" not in names.values()


def test_identical_clusters_still_get_unique_names():
    # A degenerate tie (identical vocabulary distribution in both
    # clusters) must still resolve to distinct folder names via a
    # numeric suffix, rather than silently colliding.
    token_lists = [
        ["猫", "犬"],
        ["猫", "犬"],
    ]
    labels = np.array([0, 1])

    names = assign_cluster_names(token_lists, labels)

    assert names[0] != names[1]


def test_sanitize_folder_name_strips_invalid_windows_characters():
    assert sanitize_folder_name('猫/犬:test*"<>|?') == "猫犬test"


def test_sanitize_folder_name_keeps_normal_japanese_text():
    assert sanitize_folder_name("結婚式") == "結婚式"
