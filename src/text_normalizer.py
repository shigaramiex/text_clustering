import re

import jaconv

_WHITESPACE_RE = re.compile(r"[ 　\t\r\n]+")

_ZENKAKU_ALNUM = (
    "０１２３４５６７８９"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
)
_HANKAKU_ALNUM = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)
_ZEN_TO_HAN_TABLE = str.maketrans(_ZENKAKU_ALNUM, _HANKAKU_ALNUM)


def normalize_text(text: str) -> str:
    """英数字は半角に、かなは全角に変換し、その後すべての空白を除去する。
    記号・句読点の全角/半角は変更しない。処理はメモリ上でのみ行う。
    """
    text = jaconv.h2z(text, kana=True, ascii=False, digit=False)
    text = text.translate(_ZEN_TO_HAN_TABLE)
    return _WHITESPACE_RE.sub("", text)
