from pathlib import Path
from typing import Callable

from src.clusterer import cluster_documents
from src.embedder import embed_texts
from src.file_organizer import organize_files
from src.namer import assign_cluster_names
from src.text_normalizer import normalize_text
from src.tokenizer import extract_nouns

# The livedoor news corpus format: line 1 = URL, line 2 = timestamp,
# the rest is the title and body, which is the actual article content.
_METADATA_LINE_COUNT = 2


def read_article_text(path: Path) -> str:
    """Read an article file and drop its URL/timestamp header lines."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[_METADATA_LINE_COUNT:])


def prepare_document(path: Path) -> list[str]:
    """Read, normalize and tokenize an article into content-bearing nouns."""
    raw_text = read_article_text(path)
    normalized = normalize_text(raw_text)
    return extract_nouns(normalized)


def process_genre_folder(
    genre_dir: Path,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    embed_fn: Callable = embed_texts,
    cluster_fn: Callable = cluster_documents,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Sub-cluster the articles inside one genre folder and move each file
    into a subfolder named after its cluster's representative noun.
    """
    genre_dir = Path(genre_dir)

    def report(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    file_paths = sorted(p for p in genre_dir.iterdir() if p.is_file() and p.suffix == ".txt")
    report(f"{genre_dir.name}: {len(file_paths)}件のファイルを検出しました")

    if len(file_paths) < 2:
        report(f"{genre_dir.name}: クラスタリングに十分なファイル数がないためスキップします")
        return {"total_files": len(file_paths), "num_clusters": 0}

    report(f"{genre_dir.name}: 前処理・形態素解析を実行中...")
    token_lists = [prepare_document(path) for path in file_paths]
    cleaned_texts = [" ".join(tokens) for tokens in token_lists]

    report(f"{genre_dir.name}: ベクトル化中...")
    vectors = embed_fn(cleaned_texts)

    report(f"{genre_dir.name}: クラスタ数を自動決定してクラスタリング中...")
    labels = cluster_fn(vectors, k_min=k_min, k_max=k_max, random_state=random_state)

    names_by_label = assign_cluster_names(token_lists, labels)
    cluster_names = [names_by_label[label] for label in labels]

    report(f"{genre_dir.name}: {len(names_by_label)}個のクラスタにファイルを振り分け中...")
    organize_files(genre_dir, file_paths, cluster_names)

    report(f"{genre_dir.name}: 完了 ({len(names_by_label)}クラスタ)")
    return {"total_files": len(file_paths), "num_clusters": len(names_by_label)}
