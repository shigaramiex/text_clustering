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
    """Pick a representative noun per cluster from words that stand out in
    that cluster relative to the others.

    Using the plain top mean-TF-IDF term tends to pick words that are
    common across the whole genre (e.g. "映画" for every cluster inside a
    movie-news folder), which makes cluster names indistinguishable. This
    instead scores each term by (mean TF-IDF inside the cluster) minus
    (mean TF-IDF outside the cluster), so a term shared by every cluster
    scores near zero and loses to a term unique to one cluster.

    Colliding names (e.g. a genuine tie) are disambiguated with a numeric
    suffix.
    """
    vectorizer = TfidfVectorizer(
        tokenizer=lambda tokens: tokens,
        preprocessor=lambda tokens: tokens,
        token_pattern=None,
        lowercase=False,
    )
    tfidf = vectorizer.fit_transform(token_lists).toarray()
    terms = vectorizer.get_feature_names_out()
    labels = np.asarray(labels)

    names: dict[int, str] = {}
    used_names: dict[str, int] = {}
    for label in sorted(set(labels)):
        in_cluster = labels == label
        out_cluster = ~in_cluster

        in_cluster_mean = tfidf[in_cluster].mean(axis=0)
        out_cluster_mean = (
            tfidf[out_cluster].mean(axis=0)
            if out_cluster.any()
            else np.zeros_like(in_cluster_mean)
        )
        distinctiveness = in_cluster_mean - out_cluster_mean

        top_term = sanitize_folder_name(terms[distinctiveness.argmax()])

        count = used_names.get(top_term, 0) + 1
        used_names[top_term] = count
        names[label] = top_term if count == 1 else f"{top_term}_{count}"

    return names
