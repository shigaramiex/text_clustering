import shutil
from pathlib import Path


def organize_files(
    genre_dir: Path, file_paths: list[Path], cluster_names: list[str]
) -> dict[Path, Path]:
    """Move each file into a subfolder of genre_dir named after its cluster.

    Files are moved as-is (shutil.move preserves bytes); their content is
    never read or rewritten by this function.
    """
    genre_dir = Path(genre_dir)
    moved: dict[Path, Path] = {}

    for file_path, cluster_name in zip(file_paths, cluster_names):
        cluster_dir = genre_dir / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)
        destination = cluster_dir / file_path.name
        shutil.move(str(file_path), str(destination))
        moved[file_path] = destination

    return moved
