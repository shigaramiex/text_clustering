from sudachipy import Dictionary, SplitMode

# それ単体では話題としての意味を持たない汎用的な名詞。
# 除外しないとあらゆるジャンルでクラスタ名・代表語を占拠してしまう。
STOPWORD_NOUNS = frozenset(
    {
        "こと",
        "もの",
        "ところ",
        "ため",
        "よう",
        "とき",
        "ところ",
        "こと",
        "もの",
        "こと", 
    }
)

_tokenizer_obj = Dictionary().create()
_SPLIT_MODE = SplitMode.C


def extract_nouns(text: str) -> list[str]:
    """テキストを分かち書きし、内容を表す名詞トークンのみを返す。

    助詞・助動詞・数詞・代名詞・記号、および汎用的なストップワード名詞は
    除外され、クラスタリングや代表語抽出に有用なトークンのみが残る。
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
