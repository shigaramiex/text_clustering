from pathlib import Path
from typing import Callable

from src.clusterer import cluster_documents
from src.embedder import embed_texts
from src.file_organizer import copy_files_into_clusters, output_dir_for
from src.labeler import (
    compute_ctfidf_keywords,
    compute_document_frequency_ratios,
    find_representative_titles,
    resolve_cluster_names,
)
from src.text_normalizer import normalize_text
from src.tokenizer import extract_nouns

# livedoorニュースコーパスの形式: 1行目=URL、2行目=日時、
# それ以降がタイトル・本文（実際の記事内容）。
_METADATA_LINE_COUNT = 2
_TITLE_LINE_INDEX = 0

# クラス名の選定は、ユーザーに表示する件数よりも深い候補プールを探索する。
# 非常に粗いクラスタリングでは、上位数語のc-TF-IDF候補が
# すべて汎用的・突出した語になってしまい、表示件数分の
# リストの中に非突出語が1つも残らないことがあるため。
_KEYWORD_SEARCH_POOL_SIZE = 20


def read_article_text(path: Path) -> str:
    """記事ファイルを読み込み、URL・日時のヘッダー行を除いて返す。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[_METADATA_LINE_COUNT:])


def read_article_title(path: Path) -> str:
    """記事のタイトル行のみを読み込む（URL・日時ヘッダーの次の行）。"""
    return read_article_text(path).splitlines()[_TITLE_LINE_INDEX]


def prepare_document(path: Path) -> list[str]:
    """記事を読み込み、正規化・分かち書きして内容を表す名詞を返す。"""
    raw_text = read_article_text(path)
    normalized = normalize_text(raw_text)
    return extract_nouns(normalized)


def process_genre_folder(
    genre_dir: Path,
    k_min: int = 20,
    k_max: int = 30,
    random_state: int = 42,
    fixed_k: int | None = None,
    top_n_keywords: int = 20,
    top_n_representatives: int = 3,
    embed_fn: Callable = embed_texts,
    cluster_fn: Callable = cluster_documents,
    keyword_fn: Callable = compute_ctfidf_keywords,
    document_freq_fn: Callable = compute_document_frequency_ratios,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """1つのジャンルフォルダ内の記事をサブクラスタリングし、各ファイルを
    （クラスタの上位c-TF-IDFキーワードにちなんだ名前の）サブフォルダへ
    コピーする。コピー先は同階層の新しい出力フォルダで、元のフォルダは
    一切変更しない。

    クラスタ名はクラスベースTF-IDF（BERTopicで使われるc-TF-IDF）で
    決定する: クラスタに属する記事をすべて連結して1つの単語の袋に
    まとめ、TF(t,cluster) * IDF(t) をクラスタ間で計算することで、
    全クラスタに共通する語（例: 映画ジャンルでの「映画」）は
    1つのクラスタに集中する語より優先度が下がる。
    各クラスタには、重心とのコサイン類似度が近い代表記事タイトルも
    いくつか付与される。
    """
    genre_dir = Path(genre_dir)

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    file_paths = sorted(p for p in genre_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    report(f"{genre_dir.name}: {len(file_paths)}件のファイルを検出しました")

    if len(file_paths) < 2:
        report(f"{genre_dir.name}: クラスタリングに十分なファイル数がないためスキップします")
        return {
            "total_files": len(file_paths),
            "num_clusters": 0,
            "output_dir": None,
            "keywords": {},
            "representative_titles": {},
        }

    report(f"{genre_dir.name}: 前処理・形態素解析を実行中...")
    token_lists = [prepare_document(path) for path in file_paths]
    titles = [read_article_title(path) for path in file_paths]
    cleaned_texts = [" ".join(tokens) for tokens in token_lists]

    report(f"{genre_dir.name}: ベクトル化中...")
    vectors = embed_fn(cleaned_texts)

    if fixed_k is not None:
        report(f"{genre_dir.name}: 指定されたクラスタ数 k={fixed_k} でクラスタリング中...")
    else:
        report(f"{genre_dir.name}: クラスタ数を自動決定してクラスタリング中...")
    labels = cluster_fn(
        vectors, k_min=k_min, k_max=k_max, random_state=random_state, fixed_k=fixed_k
    )

    search_pool_size = max(top_n_keywords, _KEYWORD_SEARCH_POOL_SIZE)
    resolution_keywords_by_label = keyword_fn(token_lists, labels, top_n=search_pool_size)
    document_freq_ratios = document_freq_fn(token_lists)
    names_by_label = resolve_cluster_names(resolution_keywords_by_label, document_freq_ratios)
    representatives_by_label = find_representative_titles(
        vectors, labels, titles, top_n=top_n_representatives
    )
    cluster_names = [names_by_label[label] for label in labels]
    display_keywords_by_label = {
        label: keywords[:top_n_keywords]
        for label, keywords in resolution_keywords_by_label.items()
    }

    output_dir = output_dir_for(genre_dir)
    report(
        f"{genre_dir.name}: {len(names_by_label)}個のクラスタに分けて"
        f"{output_dir.name} へコピー中..."
    )
    copy_files_into_clusters(output_dir, file_paths, cluster_names)

    report(f"{genre_dir.name}: 完了 ({len(names_by_label)}クラスタ, 出力先: {output_dir})")
    return {
        "total_files": len(file_paths),
        "num_clusters": len(names_by_label),
        "output_dir": output_dir,
        "keywords": {
            names_by_label[label]: keywords
            for label, keywords in display_keywords_by_label.items()
        },
        "representative_titles": {
            names_by_label[label]: titles_
            for label, titles_ in representatives_by_label.items()
        },
    }
