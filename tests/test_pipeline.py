import inspect

import numpy as np

from src.clusterer import DEFAULT_K_MAX, DEFAULT_K_MIN
from src.labeler import DEFAULT_TOP_N_KEYWORDS, DEFAULT_TOP_N_REPRESENTATIVES
from src.pipeline import (
    prepare_document,
    process_genre_folder,
    read_article_text,
    read_article_title,
)


def test_process_genre_folder_defaults_match_the_shared_constants():
    # Guards against pipeline.py's parameter defaults silently drifting
    # away from clusterer.py's / labeler.py's single source of truth.
    params = inspect.signature(process_genre_folder).parameters
    assert params["k_min"].default == DEFAULT_K_MIN
    assert params["k_max"].default == DEFAULT_K_MAX
    assert params["top_n_keywords"].default == DEFAULT_TOP_N_KEYWORDS
    assert params["top_n_representatives"].default == DEFAULT_TOP_N_REPRESENTATIVES


SAMPLE_ARTICLE = (
    "http://news.livedoor.com/article/detail/1234567/\n"
    "2010-05-22T14:30:00+0900\n"
    "友人代表のスピーチについて\n"
    "\n"
    "　結婚式の友人代表スピーチはとても緊張するものだ。\n"
)


def test_read_article_text_skips_url_and_timestamp_lines(tmp_path):
    article = tmp_path / "article.txt"
    article.write_text(SAMPLE_ARTICLE, encoding="utf-8")

    text = read_article_text(article)

    assert "http://" not in text
    assert "2010-05-22" not in text
    assert "友人代表のスピーチについて" in text
    assert "結婚式の友人代表スピーチ" in text


def test_read_article_title_returns_just_the_title_line(tmp_path):
    article = tmp_path / "article.txt"
    article.write_text(SAMPLE_ARTICLE, encoding="utf-8")

    assert read_article_title(article) == "友人代表のスピーチについて"


def test_prepare_document_returns_normalized_noun_tokens(tmp_path):
    article = tmp_path / "article.txt"
    article.write_text(SAMPLE_ARTICLE, encoding="utf-8")

    tokens = prepare_document(article)

    assert "スピーチ" in tokens
    assert "結婚式" in tokens
    # 助詞・助動詞は残ってはいけない
    assert "の" not in tokens
    assert "は" not in tokens


def _fake_embed_fn(texts):
    # 決定論的なフェイク埋め込み: クリーニング済みテキストに「猫」か
    # 「サッカー」が含まれるかでクラスタ分けする。これにより実際の
    # sentence-transformersモデルを読み込まずにパイプラインの
    # ロジックをテストできる。
    vectors = []
    for text in texts:
        if "猫" in text:
            vectors.append([10.0, 0.0])
        else:
            vectors.append([0.0, 10.0])
    return np.array(vectors)


def _fake_cluster_fn(vectors, k_min, k_max, random_state, fixed_k=None):
    # _fake_embed_fnが作る2つの分離したグループを分ける単純なクラスタリング。
    return np.array([0 if v[0] > v[1] else 1 for v in vectors])


def _write_article(path, title, body):
    path.write_text(
        "http://example.com/1/\n2020-01-01T00:00:00+0900\n" f"{title}\n\n{body}\n",
        encoding="utf-8",
    )


def test_process_genre_folder_copies_files_into_two_named_clusters(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    _write_article(genre_dir / "a.txt", "猫の記事", "猫と犬について猫は可愛い")
    _write_article(genre_dir / "b.txt", "猫の記事2", "猫の飼い方猫の餌")
    _write_article(genre_dir / "c.txt", "サッカーの記事", "サッカーの試合サッカー選手")
    _write_article(genre_dir / "d.txt", "サッカーの記事2", "サッカーの得点サッカー観戦")

    summary = process_genre_folder(
        genre_dir,
        k_min=2,
        k_max=2,
        embed_fn=_fake_embed_fn,
        cluster_fn=_fake_cluster_fn,
    )

    output_dir = tmp_path / "dokujo-tsushin_clustered"
    subfolders = {p.name for p in output_dir.iterdir() if p.is_dir()}
    assert len(subfolders) == 2
    assert summary["total_files"] == 4
    assert summary["num_clusters"] == 2
    assert summary["output_dir"] == output_dir
    # 元のフォルダとファイルは完全に変更されていない
    assert (genre_dir / "a.txt").exists()
    assert {p.name for p in genre_dir.iterdir()} == {"a.txt", "b.txt", "c.txt", "d.txt"}


def test_process_genre_folder_returns_keywords_and_representative_titles(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    _write_article(genre_dir / "a.txt", "猫の記事", "猫と犬について猫は可愛い")
    _write_article(genre_dir / "b.txt", "猫の記事2", "猫の飼い方猫の餌")
    _write_article(genre_dir / "c.txt", "サッカーの記事", "サッカーの試合サッカー選手")
    _write_article(genre_dir / "d.txt", "サッカーの記事2", "サッカーの得点サッカー観戦")

    summary = process_genre_folder(
        genre_dir,
        k_min=2,
        k_max=2,
        embed_fn=_fake_embed_fn,
        cluster_fn=_fake_cluster_fn,
    )

    output_dir = tmp_path / "dokujo-tsushin_clustered"
    folder_names = {p.name for p in output_dir.iterdir() if p.is_dir()}

    assert set(summary["keywords"].keys()) == folder_names
    assert set(summary["representative_titles"].keys()) == folder_names
    for name in folder_names:
        assert isinstance(summary["keywords"][name], list)
        assert isinstance(summary["representative_titles"][name], list)
        assert len(summary["representative_titles"][name]) > 0


def test_process_genre_folder_skips_when_fewer_than_two_files(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    _write_article(genre_dir / "a.txt", "猫の記事", "猫について")

    summary = process_genre_folder(
        genre_dir,
        k_min=2,
        k_max=2,
        embed_fn=_fake_embed_fn,
        cluster_fn=_fake_cluster_fn,
    )

    assert summary["total_files"] == 1
    assert summary["num_clusters"] == 0
    assert (genre_dir / "a.txt").exists()
    assert summary["keywords"] == {}
    assert summary["representative_titles"] == {}


def test_process_genre_folder_reports_progress(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    _write_article(genre_dir / "a.txt", "猫の記事", "猫について猫")
    _write_article(genre_dir / "b.txt", "サッカーの記事", "サッカーについてサッカー")

    messages = []
    process_genre_folder(
        genre_dir,
        k_min=2,
        k_max=2,
        embed_fn=_fake_embed_fn,
        cluster_fn=_fake_cluster_fn,
        progress_callback=messages.append,
    )

    assert len(messages) > 0


def test_process_genre_folder_passes_fixed_k_to_cluster_fn(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    _write_article(genre_dir / "a.txt", "猫の記事", "猫について猫")
    _write_article(genre_dir / "b.txt", "サッカーの記事", "サッカーについてサッカー")

    received_kwargs = {}

    def spy_cluster_fn(vectors, k_min, k_max, random_state, fixed_k=None):
        received_kwargs["fixed_k"] = fixed_k
        return _fake_cluster_fn(vectors, k_min, k_max, random_state, fixed_k)

    process_genre_folder(
        genre_dir,
        fixed_k=7,
        embed_fn=_fake_embed_fn,
        cluster_fn=spy_cluster_fn,
    )

    assert received_kwargs["fixed_k"] == 7


def test_process_genre_folder_requests_a_larger_keyword_pool_than_it_displays(tmp_path):
    # 名前解決の探索プールは、top_n_keywords（実際にユーザーへ表示する件数）
    # より大きくなければならない。上位数件の候補がすべて突出/汎用的な語
    # だった場合でも、ランクの下位に本当に特徴的な語を見つける余地を
    # 残すため。
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    _write_article(genre_dir / "a.txt", "猫の記事", "猫について猫")
    _write_article(genre_dir / "b.txt", "サッカーの記事", "サッカーについてサッカー")

    received_kwargs = {}

    def spy_keyword_fn(token_lists, labels, top_n=5):
        received_kwargs["top_n"] = top_n
        return {label: ["キーワード"] for label in set(labels)}

    process_genre_folder(
        genre_dir,
        top_n_keywords=3,
        embed_fn=_fake_embed_fn,
        cluster_fn=_fake_cluster_fn,
        keyword_fn=spy_keyword_fn,
    )

    assert received_kwargs["top_n"] > 3


def test_process_genre_folder_displays_only_top_n_keywords_even_from_a_larger_pool(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    _write_article(genre_dir / "a.txt", "猫の記事", "猫について猫")
    _write_article(genre_dir / "b.txt", "サッカーの記事", "サッカーについてサッカー")

    def fake_keyword_fn(token_lists, labels, top_n=5):
        return {label: [f"語{i}" for i in range(top_n)] for label in set(labels)}

    summary = process_genre_folder(
        genre_dir,
        top_n_keywords=3,
        embed_fn=_fake_embed_fn,
        cluster_fn=_fake_cluster_fn,
        keyword_fn=fake_keyword_fn,
    )

    for keywords in summary["keywords"].values():
        assert len(keywords) == 3


def test_process_genre_folder_avoids_all_dominant_collapse_by_searching_deeper(tmp_path):
    # 回帰テスト: 非常に粗いクラスタリングでは、クラスタの上位数件の
    # c-TF-IDF候補が全て汎用的・突出した語になってしまい、表示用の
    # 小さいリストの中には非突出な選択肢が1つも残らないことがある。
    # 名前解決はより大きな候補プールを探索し、ランクの下位にある
    # （より大きなtop_nを要求したときのみ現れる）本当に特徴的な語を
    # フォルダ名として使えるようにしなければならない。
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    _write_article(genre_dir / "a.txt", "猫の記事", "猫について猫")
    _write_article(genre_dir / "b.txt", "サッカーの記事", "サッカーについてサッカー")

    def fake_keyword_fn(token_lists, labels, top_n=5):
        keywords = {}
        for label in sorted(set(labels)):
            filler = [f"共通語{i}" for i in range(5)]
            distinguishing = [f"特徴{label}"] if top_n > 5 else []
            keywords[label] = filler + distinguishing
        return keywords

    def fake_document_freq_fn(token_lists):
        return {f"共通語{i}": 0.9 for i in range(5)}

    summary = process_genre_folder(
        genre_dir,
        top_n_keywords=3,
        embed_fn=_fake_embed_fn,
        cluster_fn=_fake_cluster_fn,
        keyword_fn=fake_keyword_fn,
        document_freq_fn=fake_document_freq_fn,
    )

    output_dir = tmp_path / "dokujo-tsushin_clustered"
    folder_names = {p.name for p in output_dir.iterdir() if p.is_dir()}
    assert folder_names == {"特徴0", "特徴1"}
