import hashlib

from src.file_organizer import copy_files_into_clusters, output_dir_for


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_output_dir_for_is_a_sibling_of_the_source_folder(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()

    output_dir = output_dir_for(genre_dir)

    assert output_dir.parent == tmp_path
    assert output_dir != genre_dir
    assert "dokujo-tsushin" in output_dir.name


def test_files_are_copied_into_named_cluster_subfolders(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    file_a = genre_dir / "dokujo-tsushin-1.txt"
    file_b = genre_dir / "dokujo-tsushin-2.txt"
    file_a.write_bytes("結婚式のスピーチについて".encode("utf-8"))
    file_b.write_bytes("サッカーの試合について".encode("utf-8"))
    output_dir = output_dir_for(genre_dir)

    result = copy_files_into_clusters(
        output_dir,
        file_paths=[file_a, file_b],
        cluster_names=["結婚式", "サッカー"],
    )

    assert (output_dir / "結婚式" / "dokujo-tsushin-1.txt").exists()
    assert (output_dir / "サッカー" / "dokujo-tsushin-2.txt").exists()
    assert result[file_a] == output_dir / "結婚式" / "dokujo-tsushin-1.txt"
    assert result[file_b] == output_dir / "サッカー" / "dokujo-tsushin-2.txt"


def test_original_files_and_folder_are_left_untouched(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    file_a = genre_dir / "dokujo-tsushin-1.txt"
    file_a.write_bytes("結婚式のスピーチについて".encode("utf-8"))
    output_dir = output_dir_for(genre_dir)

    copy_files_into_clusters(output_dir, file_paths=[file_a], cluster_names=["結婚式"])

    assert file_a.exists()
    assert list(genre_dir.iterdir()) == [file_a]


def test_copied_file_content_is_byte_for_byte_unchanged(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    file_a = genre_dir / "dokujo-tsushin-1.txt"
    original_bytes = "友人代表のスピーチ、独女（３３歳）は\n\nどうこなしている？".encode(
        "utf-8"
    )
    file_a.write_bytes(original_bytes)
    original_hash = _sha256(file_a)
    output_dir = output_dir_for(genre_dir)

    result = copy_files_into_clusters(
        output_dir, file_paths=[file_a], cluster_names=["結婚式"]
    )

    copied_path = result[file_a]
    assert _sha256(copied_path) == original_hash
    assert copied_path.read_bytes() == original_bytes
    # source is untouched too
    assert _sha256(file_a) == original_hash


def test_multiple_files_in_same_cluster_share_one_subfolder(tmp_path):
    genre_dir = tmp_path / "dokujo-tsushin"
    genre_dir.mkdir()
    file_a = genre_dir / "a.txt"
    file_b = genre_dir / "b.txt"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")
    output_dir = output_dir_for(genre_dir)

    copy_files_into_clusters(
        output_dir, file_paths=[file_a, file_b], cluster_names=["結婚式", "結婚式"]
    )

    cluster_dir = output_dir / "結婚式"
    assert (cluster_dir / "a.txt").exists()
    assert (cluster_dir / "b.txt").exists()
