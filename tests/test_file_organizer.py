import hashlib

from src.file_organizer import organize_files


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_files_are_moved_into_named_cluster_subfolders(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    file_a = genre_dir / "dokujo-tsushin-1.txt"
    file_b = genre_dir / "dokujo-tsushin-2.txt"
    file_a.write_bytes("結婚式のスピーチについて".encode("utf-8"))
    file_b.write_bytes("サッカーの試合について".encode("utf-8"))

    result = organize_files(
        genre_dir,
        file_paths=[file_a, file_b],
        cluster_names=["結婚式", "サッカー"],
    )

    assert (genre_dir / "結婚式" / "dokujo-tsushin-1.txt").exists()
    assert (genre_dir / "サッカー" / "dokujo-tsushin-2.txt").exists()
    assert result[file_a] == genre_dir / "結婚式" / "dokujo-tsushin-1.txt"
    assert result[file_b] == genre_dir / "サッカー" / "dokujo-tsushin-2.txt"


def test_original_files_no_longer_exist_at_old_path(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    file_a = genre_dir / "dokujo-tsushin-1.txt"
    file_a.write_bytes("結婚式のスピーチについて".encode("utf-8"))

    organize_files(genre_dir, file_paths=[file_a], cluster_names=["結婚式"])

    assert not file_a.exists()


def test_file_content_is_byte_for_byte_unchanged(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    file_a = genre_dir / "dokujo-tsushin-1.txt"
    original_bytes = "友人代表のスピーチ、独女（３３歳）は\n\nどうこなしている？".encode(
        "utf-8"
    )
    file_a.write_bytes(original_bytes)
    original_hash = _sha256(file_a)

    result = organize_files(genre_dir, file_paths=[file_a], cluster_names=["結婚式"])

    moved_path = result[file_a]
    assert _sha256(moved_path) == original_hash
    assert moved_path.read_bytes() == original_bytes


def test_multiple_files_in_same_cluster_share_one_subfolder(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    file_a = genre_dir / "a.txt"
    file_b = genre_dir / "b.txt"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")

    organize_files(
        genre_dir, file_paths=[file_a, file_b], cluster_names=["結婚式", "結婚式"]
    )

    cluster_dir = genre_dir / "結婚式"
    assert (cluster_dir / "a.txt").exists()
    assert (cluster_dir / "b.txt").exists()
