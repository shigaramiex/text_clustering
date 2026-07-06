from src.namer import sanitize_folder_name


def test_sanitize_folder_name_strips_invalid_windows_characters():
    assert sanitize_folder_name('猫/犬:test*"<>|?') == "猫犬test"


def test_sanitize_folder_name_keeps_normal_japanese_text():
    assert sanitize_folder_name("結婚式") == "結婚式"
