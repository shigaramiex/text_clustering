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
    """Convert ASCII letters/digits to half-width and kana to
    full-width, then strip all whitespace. Punctuation/symbol width is
    left untouched. Operates purely in memory.
    """
    text = jaconv.h2z(text, kana=True, ascii=False, digit=False)
    text = text.translate(_ZEN_TO_HAN_TABLE)
    return _WHITESPACE_RE.sub("", text)
