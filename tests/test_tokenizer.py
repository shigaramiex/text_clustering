from src.tokenizer import extract_nouns


def test_particles_are_excluded():
    tokens = extract_nouns("友人代表のスピーチを行った")
    assert "の" not in tokens
    assert "を" not in tokens


def test_numbers_are_excluded():
    tokens = extract_nouns("3回行った")
    assert "3" not in tokens


def test_verbs_and_auxiliary_verbs_are_excluded():
    tokens = extract_nouns("買った")
    assert "買う" not in tokens
    assert "た" not in tokens


def test_nouns_are_kept_with_normalized_form():
    tokens = extract_nouns("友人代表のスピーチを東京都渋谷区で行った")
    assert "友人" in tokens
    assert "代表" in tokens
    assert "スピーチ" in tokens
    assert "東京都渋谷区" in tokens


def test_pronouns_are_excluded():
    tokens = extract_nouns("これは私のかばんです")
    assert "此れ" not in tokens
    assert "私" not in tokens


def test_generic_stopword_nouns_are_excluded():
    tokens = extract_nouns("これはことのためによい")
    assert "こと" not in tokens


def test_punctuation_is_excluded():
    tokens = extract_nouns("友人代表。スピーチ、独女")
    assert "。" not in tokens
    assert "、" not in tokens


def test_returns_empty_list_for_empty_string():
    assert extract_nouns("") == []
