import shutil
from pathlib import Path

OUTPUT_DIR_SUFFIX = "_clustered"


def output_dir_for(source_dir: Path) -> Path:
    """source_dirのクラスタリング結果を出力する、同階層の出力フォルダ。"""
    source_dir = Path(source_dir)
    return source_dir.parent / f"{source_dir.name}{OUTPUT_DIR_SUFFIX}"


def copy_files_into_clusters(
    output_dir: Path, file_paths: list[Path], cluster_names: list[str]
) -> dict[Path, Path]:
    """各ファイルをoutput_dir配下のクラスタサブフォルダへコピーする。

    元ファイルは読み込むだけで、移動や書き換えは一切行わないため、
    元のフォルダは完全に変更されずそのまま残る。
    """
    output_dir = Path(output_dir)
    copied: dict[Path, Path] = {}

    for file_path, cluster_name in zip(file_paths, cluster_names):
        cluster_dir = output_dir / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)
        destination = cluster_dir / file_path.name
        shutil.copy2(str(file_path), str(destination))
        copied[file_path] = destination

    return copied
