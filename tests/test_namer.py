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


def test_duplicate_top_terms_get_unique_suffix():
    # Both clusters are dominated by the same term "猫"; the second
    # occurrence must be disambiguated rather than silently colliding.
    token_lists = [
        ["猫", "猫", "犬"],
        ["猫", "猫", "鳥"],
    ]
    labels = np.array([0, 1])

    names = assign_cluster_names(token_lists, labels)

    assert names[0] != names[1]
    assert names[0] == "猫"
    assert names[1] == "猫_2"


def test_sanitize_folder_name_strips_invalid_windows_characters():
    assert sanitize_folder_name('猫/犬:test*"<>|?') == "猫犬test"


def test_sanitize_folder_name_keeps_normal_japanese_text():
    assert sanitize_folder_name("結婚式") == "結婚式"
