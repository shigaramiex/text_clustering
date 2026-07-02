import shutil
from pathlib import Path

OUTPUT_DIR_SUFFIX = "_clustered"


def output_dir_for(source_dir: Path) -> Path:
    """Sibling output folder for source_dir's clustering results."""
    source_dir = Path(source_dir)
    return source_dir.parent / f"{source_dir.name}{OUTPUT_DIR_SUFFIX}"


def copy_files_into_clusters(
    output_dir: Path, file_paths: list[Path], cluster_names: list[str]
) -> dict[Path, Path]:
    """Copy each file into a cluster subfolder under output_dir.

    The source files are only ever read, never moved or rewritten, so the
    original folder is left completely untouched.
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
