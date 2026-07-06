import re

_INVALID_WINDOWS_CHARS_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_folder_name(name: str) -> str:
    """Windowsのフォルダ名として使用できない文字を除去する。"""
    return _INVALID_WINDOWS_CHARS_RE.sub("", name)
