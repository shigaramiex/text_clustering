import numpy as np

from src.pipeline import process_genre_folder, read_article_text, prepare_document


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


def test_prepare_document_returns_normalized_noun_tokens(tmp_path):
    article = tmp_path / "article.txt"
    article.write_text(SAMPLE_ARTICLE, encoding="utf-8")

    tokens = prepare_document(article)

    assert "スピーチ" in tokens
    assert "結婚式" in tokens
    # particles/auxiliary words must not leak through
    assert "の" not in tokens
    assert "は" not in tokens


def _fake_embed_fn(texts):
    # Deterministic fake embedding: cluster by whether "猫" or "サッカー"
    # appears in the cleaned text, so the pipeline logic can be tested
    # without loading the real sentence-transformers model.
    vectors = []
    for text in texts:
        if "猫" in text:
            vectors.append([10.0, 0.0])
        else:
            vectors.append([0.0, 10.0])
    return np.array(vectors)


def _fake_cluster_fn(vectors, k_min, k_max, random_state):
    # Trivial clustering matching _fake_embed_fn's two well-separated groups.
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
    # the original folder and its files are completely untouched
    assert (genre_dir / "a.txt").exists()
    assert {p.name for p in genre_dir.iterdir()} == {"a.txt", "b.txt", "c.txt", "d.txt"}


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
