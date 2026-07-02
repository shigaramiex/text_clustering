import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_INVALID_WINDOWS_CHARS_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_folder_name(name: str) -> str:
    """Strip characters that are illegal in Windows folder names."""
    return _INVALID_WINDOWS_CHARS_RE.sub("", name)


def assign_cluster_names(
    token_lists: list[list[str]], labels: np.ndarray
) -> dict[int, str]:
    """Pick the top mean-TF-IDF noun per cluster as its representative name.

    Colliding names (two clusters sharing the same top noun) are
    disambiguated with a numeric suffix.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=lambda tokens: tokens,
        preprocessor=lambda tokens: tokens,
        token_pattern=None,
        lowercase=False,
    )
    tfidf = vectorizer.fit_transform(token_lists)
    terms = vectorizer.get_feature_names_out()
    labels = np.asarray(labels)

    names: dict[int, str] = {}
    used_names: dict[str, int] = {}
    for label in sorted(set(labels)):
        mask = labels == label
        mean_scores = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_term = sanitize_folder_name(terms[mean_scores.argmax()])

        count = used_names.get(top_term, 0) + 1
        used_names[top_term] = count
        names[label] = top_term if count == 1 else f"{top_term}_{count}"

    return names
