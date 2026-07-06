import inspect
import math

import numpy as np
import pytest

from src.labeler import (
    DEFAULT_TOP_N_KEYWORDS,
    DEFAULT_TOP_N_REPRESENTATIVES,
    compute_ctfidf_keywords,
    compute_document_frequency_ratios,
    find_representative_titles,
    resolve_cluster_names,
)


def test_compute_ctfidf_keywords_default_matches_the_shared_constant():
    params = inspect.signature(compute_ctfidf_keywords).parameters
    assert params["top_n"].default == DEFAULT_TOP_N_KEYWORDS


def test_find_representative_titles_default_matches_the_shared_constant():
    params = inspect.signature(find_representative_titles).parameters
    assert params["top_n"].default == DEFAULT_TOP_N_REPRESENTATIVES


def test_returns_keywords_for_every_unique_label():
    token_lists = [["猫", "犬"], ["サッカー", "野球"], ["旅行", "電車"]]
    labels = np.array([0, 1, 2])

    keywords = compute_ctfidf_keywords(token_lists, labels, top_n=5)

    assert set(keywords.keys()) == {0, 1, 2}


def test_keywords_reflect_dominant_topic_per_cluster():
    token_lists = [
        ["猫", "犬", "動物", "散歩"],
        ["犬", "猫", "動物", "餌"],
        ["猫", "動物", "毛"],
        ["サッカー", "野球", "試合", "選手"],
        ["野球", "試合", "監督"],
        ["サッカー", "試合", "得点"],
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])

    keywords = compute_ctfidf_keywords(token_lists, labels, top_n=3)

    assert set(keywords[0]) <= {"猫", "犬", "動物", "散歩", "餌", "毛"}
    assert set(keywords[1]) <= {"サッカー", "野球", "試合", "選手", "監督", "得点"}
    assert not set(keywords[0]) & {"サッカー", "野球", "選手", "監督", "得点"}


def test_matches_hand_computed_ctfidf_scores():
    # cluster0の袋: ["a", "a", "b"]（3単語）; cluster1の袋: ["a", "c", "c"]
    # tf(a,0)=2/3, tf(a,1)=1/3, tf(b,0)=1/3, tf(c,1)=2/3
    # df(a)=2, df(b)=1, df(c)=1（2クラスタ中）
    # idf(a)=log(1+2/2)=log(2); idf(b)=idf(c)=log(1+2/1)=log(3)
    # score(a,0)=2/3*log(2)=0.4621, score(b,0)=1/3*log(3)=0.3662 -> cluster0は"a"が勝つ
    # score(a,1)=1/3*log(2)=0.2310, score(c,1)=2/3*log(3)=0.7324 -> cluster1は"c"が勝つ
    token_lists = [["a", "a", "b"], ["a", "c", "c"]]
    labels = np.array([0, 1])

    keywords = compute_ctfidf_keywords(token_lists, labels, top_n=2)

    assert keywords[0][0] == "a"
    assert keywords[1][0] == "c"


def test_ubiquitous_term_sinks_relative_to_distinguishing_terms():
    # 「映画」は各クラスタの袋に1回ずつ出現する（全4クラスタに存在し
    # idfが低い）一方、各クラスタ固有の語も同じ頻度でクラスタ内に
    # 出現するが他クラスタには出現しない（idfが高い）。両者のクラスタ内
    # 出現頻度が同じでも、遍在する語はスコアが低くなり、
    # どのクラスタでも1位のキーワードになってはならない。
    token_lists = [
        ["映画", "アニメ"],
        ["映画", "ハリウッド"],
        ["映画", "ホラー"],
        ["映画", "恋愛"],
    ]
    labels = np.array([0, 1, 2, 3])

    keywords = compute_ctfidf_keywords(token_lists, labels, top_n=5)

    assert keywords[0][0] == "アニメ"
    assert keywords[1][0] == "ハリウッド"
    assert keywords[2][0] == "ホラー"
    assert keywords[3][0] == "恋愛"
    for label, own_term in zip([0, 1, 2, 3], ["アニメ", "ハリウッド", "ホラー", "恋愛"]):
        assert keywords[label].index(own_term) < (
            keywords[label].index("映画") if "映画" in keywords[label] else math.inf
        )


def test_top_n_limits_the_number_of_keywords_returned():
    token_lists = [["a", "b", "c", "d", "e", "f"]]
    labels = np.array([0])

    keywords = compute_ctfidf_keywords(token_lists, labels, top_n=3)

    assert len(keywords[0]) == 3


def test_handles_cluster_with_no_tokens_without_crashing():
    token_lists = [[], ["猫", "犬"]]
    labels = np.array([0, 1])

    keywords = compute_ctfidf_keywords(token_lists, labels, top_n=5)

    assert keywords[0] == []
    assert "猫" in keywords[1] or "犬" in keywords[1]


def test_document_frequency_ratios_reflect_share_of_all_documents():
    # 「映画」は4文書中3文書に出現し、「アニメ」は1文書のみ。
    token_lists = [
        ["映画", "アニメ"],
        ["映画"],
        ["映画"],
        ["恋愛"],
    ]

    ratios = compute_document_frequency_ratios(token_lists)

    assert ratios["映画"] == pytest.approx(0.75)
    assert ratios["アニメ"] == pytest.approx(0.25)
    assert ratios["恋愛"] == pytest.approx(0.25)


def test_document_frequency_ratios_ignore_repeats_within_one_document():
    # 「映画」が1つの文書内で10回繰り返されても、2文書中1文書としてしか数えない。
    token_lists = [["映画"] * 10, ["アニメ"]]

    ratios = compute_document_frequency_ratios(token_lists)

    assert ratios["映画"] == pytest.approx(0.5)


def test_resolve_cluster_names_picks_top_keyword():
    keywords_by_label = {0: ["猫", "犬"], 1: ["サッカー", "野球"]}
    document_freq_ratios = {"猫": 0.2, "犬": 0.1, "サッカー": 0.2, "野球": 0.1}

    names = resolve_cluster_names(keywords_by_label, document_freq_ratios)

    assert names == {0: "猫", 1: "サッカー"}


def test_resolve_cluster_names_prefers_a_distinct_second_choice_over_duplicating():
    # 「猫」は突出した汎用語（全体の大半の文書に出現）なので、
    # 両クラスタを「猫」「猫_2」と命名する（同じ話題が2つあるように
    # 読める）のではなく、それぞれが自分固有の次点候補にフォール
    # スルーするべきである。
    keywords_by_label = {0: ["猫", "犬"], 1: ["猫", "鳥"]}
    document_freq_ratios = {"猫": 0.9, "犬": 0.1, "鳥": 0.1}

    names = resolve_cluster_names(keywords_by_label, document_freq_ratios)

    assert names[0] != names[1]
    assert names[0] == "犬"
    assert names[1] == "鳥"


def test_resolve_cluster_names_avoids_a_term_dominant_across_all_documents():
    # 「映画」は全クラスタの候補リストの1位であり、かつフォルダ内の
    # 大半の文書に出現するため、区別に役立つ情報を持たない。各クラスタは
    # 代わりに自分固有の2位の語にフォールスルーするべきである。
    keywords_by_label = {
        0: ["映画", "アニメ"],
        1: ["映画", "ホラー"],
        2: ["映画", "恋愛"],
    }
    document_freq_ratios = {"映画": 0.9, "アニメ": 0.1, "ホラー": 0.1, "恋愛": 0.1}

    names = resolve_cluster_names(keywords_by_label, document_freq_ratios)

    assert set(names.values()) == {"アニメ", "ホラー", "恋愛"}
    assert "映画" not in names.values()


def test_resolve_cluster_names_catches_dominance_missed_by_cluster_level_counting():
    # 回帰テスト: ある語は文書レベルでは本当に遍在している（フォルダ内の
    # 大半の記事に出現する）にもかかわらず、一部のクラスタの上位5リスト
    # にしか入らないことがある（他のクラスタではc-TF-IDFスコアが
    # より特徴的な語に負けているため）。「いくつのクラスタの上位5に
    # 入っているか」だけを数える方法ではこれを見逃してしまう。
    # 実際の文書単位の出現頻度で検知する必要がある。
    keywords_by_label = {
        0: ["岩永", "櫻井"],
        1: ["脱獄", "映画"],
        2: ["映画", "チアー"],
        3: ["恐怖", "映画"],
        4: ["バンパイア", "映画"],
        5: ["クリステン", "ホームズ"],
        6: ["ロボット", "カウボーイ"],
        7: ["キャプテン", "アメリカ"],
        8: ["ニール", "ピーター"],
        9: ["征服", "生態"],
    }
    # 「映画」は10クラスタ中4クラスタの上位5リストにしか入っていない
    # （比率0.4であり、旧来のクラスタ数ベースの指標では非突出と
    # 判定されてしまう）が、フォルダ内の全60記事中50記事（0.833）に
    # 出現している。
    document_freq_ratios = {"映画": 0.833}

    names = resolve_cluster_names(keywords_by_label, document_freq_ratios)

    assert "映画" not in names.values()
    assert names[2] == "チアー"
    assert names[4] == "バンパイア"


def test_resolve_cluster_names_still_uses_dominant_term_as_last_resort():
    # クラスタの唯一の候補が突出語の場合（他に選択肢がない場合）でも、
    # 名前がつかずに落ちてしまうのではなく、必ず名前が付与されるべきである。
    keywords_by_label = {0: ["映画"], 1: ["映画", "ホラー"]}
    document_freq_ratios = {"映画": 0.9, "ホラー": 0.1}

    names = resolve_cluster_names(keywords_by_label, document_freq_ratios)

    assert names[0]
    assert names[1]
    assert names[0] != names[1]


def test_resolve_cluster_names_sanitizes_invalid_characters():
    keywords_by_label = {0: ['猫/犬']}

    names = resolve_cluster_names(keywords_by_label, {"猫/犬": 0.1})

    assert names[0] == "猫犬"


def test_resolve_cluster_names_falls_back_when_no_keywords():
    keywords_by_label = {0: []}

    names = resolve_cluster_names(keywords_by_label, {})

    assert names[0]


def test_resolve_cluster_names_defaults_to_no_document_frequency_data():
    # document_freq_ratiosは省略可能。省略してもクラッシュせず、
    # 突出判定を単にスキップする（すべての語を非突出として扱う）。
    keywords_by_label = {0: ["猫", "犬"], 1: ["サッカー", "野球"]}

    names = resolve_cluster_names(keywords_by_label)

    assert names == {0: "猫", 1: "サッカー"}


def test_find_representative_titles_picks_closest_to_centroid():
    doc_vectors = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1])
    titles = ["猫の記事A", "猫の記事B", "微妙な記事", "サッカー記事A", "サッカー記事B"]

    representatives = find_representative_titles(doc_vectors, labels, titles, top_n=2)

    assert set(representatives[0]) == {"猫の記事A", "猫の記事B"}
    assert set(representatives[1]) == {"サッカー記事A", "サッカー記事B"}


def test_find_representative_titles_respects_top_n():
    doc_vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    labels = np.array([0, 0, 0, 0])
    titles = ["t1", "t2", "t3", "t4"]

    representatives = find_representative_titles(doc_vectors, labels, titles, top_n=2)

    assert len(representatives[0]) == 2


def test_find_representative_titles_returns_entry_for_every_label():
    doc_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    titles = ["a", "b"]

    representatives = find_representative_titles(doc_vectors, labels, titles, top_n=3)

    assert set(representatives.keys()) == {0, 1}
