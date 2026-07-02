from sudachipy import Dictionary, SplitMode

# Generic nouns that carry little topical meaning on their own and would
# otherwise dominate clusters/representative names across every genre.
STOPWORD_NOUNS = frozenset(
    {
        "こと",
        "もの",
        "よう",
        "為",
        "はず",
        "うち",
        "とき",
        "ところ",
        "わけ",
        "ほう",
        "の",
        "さん",
        "たち",
        "そう",
        "ため",
        "上",
        "中",
        "自分",
        "みんな",
        "記事",
        "写真",
    }
)

_tokenizer_obj = Dictionary().create()
_SPLIT_MODE = SplitMode.C


def extract_nouns(text: str) -> list[str]:
    """Tokenize text and return only content-bearing noun tokens.

    Particles, auxiliary verbs, numerals, pronouns, punctuation and
    generic stopword nouns are excluded, leaving the tokens useful for
    clustering and representative-noun extraction.
    """
    nouns = []
    for morpheme in _tokenizer_obj.tokenize(text, _SPLIT_MODE):
        pos = morpheme.part_of_speech()
        if pos[0] != "名詞" or pos[1] == "数詞":
            continue
        normalized = morpheme.normalized_form()
        if normalized in STOPWORD_NOUNS:
            continue
        nouns.append(normalized)
    return nouns
