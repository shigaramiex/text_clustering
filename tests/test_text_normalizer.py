from src.text_normalizer import normalize_text


def test_ascii_letters_converted_to_half_width():
    assert normalize_text("ＡＢＣａｂｃ") == "ABCabc"


def test_digits_converted_to_half_width():
    assert normalize_text("１２３") == "123"


def test_katakana_stays_full_width():
    assert normalize_text("ヒルズ") == "ヒルズ"


def test_half_width_katakana_converted_to_full_width():
    assert normalize_text("ﾃｽﾄ") == "テスト"


def test_half_width_spaces_removed():
    assert normalize_text("東京 都 渋谷区") == "東京都渋谷区"


def test_full_width_spaces_removed():
    assert normalize_text("東京　都　渋谷区") == "東京都渋谷区"


def test_newlines_and_tabs_removed():
    assert normalize_text("東京\n都\t渋谷区\r\n") == "東京都渋谷区"


def test_mixed_text_end_to_end():
    # 全角の記号・句読点（（）？＆）は日本語のかな・漢字でも英数字でも
    # ないため、意図的に幅を変換しない。
    raw = "友人代表のスピーチ、独女（３３歳）はどうこなしている？　Ａｉ＆ｕｅｏ　１２３"
    expected = "友人代表のスピーチ、独女（33歳）はどうこなしている？Ai＆ueo123"
    assert normalize_text(raw) == expected


def test_empty_string_returns_empty_string():
    assert normalize_text("") == ""
