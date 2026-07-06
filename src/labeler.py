from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.namer import sanitize_folder_name

# キーワード・代表記事の表示件数に関する唯一の情報源。
# pipeline.pyはここから値を参照し、ハードコードしない。
DEFAULT_TOP_N_KEYWORDS = 5
DEFAULT_TOP_N_REPRESENTATIVES = 3


def compute_document_frequency_ratios(token_lists: list[list[str]]) -> dict[str, float]:
    """フォルダ内の全記事のうち、各語を含む記事の割合。

    1つの記事内で何度繰り返されても1記事としてしか数えないため、
    単純なトークン数ではなく実際の文書レベルでの広がりを反映する。
    これはコーパス全体で「汎用的」な語（例: 映画ジャンルの記事の
    ほとんどに出てくる「映画」）を検知する正しい指標である。
    「いくつのクラスタがその語に言及しているか」を数える方法では、
    ある語がフォルダ全体では頻出していても、一部のクラスタでは
    c-TF-IDFスコアが他の語に負けてしまうことがあるため、過小評価
    してしまう可能性がある。
    """
    total_docs = len(token_lists)
    if total_docs == 0:
        return {}

    doc_freq: Counter[str] = Counter()
    for tokens in token_lists:
        doc_freq.update(set(tokens))

    return {term: count / total_docs for term, count in doc_freq.items()}


def compute_ctfidf_keywords(
    token_lists: list[list[str]], labels: np.ndarray, top_n: int = DEFAULT_TOP_N_KEYWORDS
) -> dict[int, list[str]]:
    """BERTopic方式のクラスベースTF-IDF（c-TF-IDF）によるキーワード抽出。

    クラスタに属する記事をすべて連結し、クラスタごとに1つの単語の袋
    （bag of words）を作った上で、以下を計算する:

        TF(t, cluster)  = クラスタ内でのtの出現回数 / クラスタの総単語数
        IDF(t)          = log(1 + 全クラスタ数 / tが出現するクラスタ数)
        score(t, cluster) = TF(t, cluster) * IDF(t)

    全クラスタに共通する語はidfが低くなるため、クラスタ内出現頻度が
    同程度でも、特定クラスタに集中する語より優先度が下がる。
    クラスタごとにスコア降順で上位top_n件のキーワードを返す
    （スコアが正の語のみを含む）。
    """
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels))

    cluster_bags = []
    for label in unique_labels:
        mask = labels == label
        bag: list[str] = []
        for tokens, included in zip(token_lists, mask):
            if included:
                bag.extend(tokens)
        cluster_bags.append(bag)

    vectorizer = CountVectorizer(
        tokenizer=lambda tokens: tokens,
        preprocessor=lambda tokens: tokens,
        token_pattern=None,
        lowercase=False,
    )
    counts = vectorizer.fit_transform(cluster_bags).toarray().astype(float)
    terms = vectorizer.get_feature_names_out()

    num_clusters = len(cluster_bags)
    cluster_word_totals = counts.sum(axis=1, keepdims=True)
    cluster_word_totals[cluster_word_totals == 0] = 1  # 空の袋でのゼロ除算を回避
    tf = counts / cluster_word_totals

    doc_freq = (counts > 0).sum(axis=0)
    doc_freq[doc_freq == 0] = 1  # 使われない語はどのみちtfが0になる
    idf = np.log(1 + num_clusters / doc_freq)

    scores = tf * idf

    keywords_by_label: dict[int, list[str]] = {}
    for row_idx, label in enumerate(unique_labels):
        row_scores = scores[row_idx]
        ranked = np.argsort(row_scores)[::-1]
        keywords_by_label[label] = [terms[i] for i in ranked[:top_n] if row_scores[i] > 0]

    return keywords_by_label


def resolve_cluster_names(
    keywords_by_label: dict[int, list[str]],
    document_freq_ratios: dict[str, float] | None = None,
    max_document_freq_ratio: float = 0.5,
) -> dict[int, str]:
    """ランク付けされたキーワード候補から、クラスタごとに重複のない
    フォルダ名を選ぶ。

    常に各クラスタの1位のキーワードを採用すると重複しやすい。
    複数のクラスタが同じ上位語（例: 映画ジャンルでの「映画」）を
    共有し、「_2」「_3」という連番だけで区別されがちで、同じ話題が
    繰り返されているように読めてしまう。そこで各クラスタは自分の
    候補リスト（c-TF-IDFスコアで既にランク付け済み）の中から、
    以下の両方を満たす最初の語を探す:
      - まだ他のクラスタに使われていない
      - 「突出」していない（document_freq_ratiosにおいて、
        フォルダ内の全記事のうちmax_document_freq_ratioを超える
        割合に出現していない = このクラスタ固有ではなく
        コーパス全体で汎用的な語である兆候）
    すべての候補が失格となった場合のみ、クラスタは残っている
    最善の語（突出している語や使用済みの語も含む）にフォールバック
    し、必要であれば連番で重複を回避する。
    """
    document_freq_ratios = document_freq_ratios or {}
    labels = sorted(keywords_by_label)

    names: dict[int, str] = {}
    used_names: set[str] = set()

    for label in labels:
        keywords = keywords_by_label[label]
        non_dominant = [
            t for t in keywords if document_freq_ratios.get(t, 0.0) <= max_document_freq_ratio
        ]
        dominant = [
            t for t in keywords if document_freq_ratios.get(t, 0.0) > max_document_freq_ratio
        ]

        chosen_name = None
        for term in non_dominant + dominant:
            sanitized = sanitize_folder_name(term) or f"クラス{label}"
            if sanitized not in used_names:
                chosen_name = sanitized
                break

        if chosen_name is None:
            # 候補プールが小さすぎて全候補が使用済みの場合:
            # 1位の語に連番を付けて重複を回避する。
            base = sanitize_folder_name(keywords[0]) if keywords else f"クラス{label}"
            base = base or f"クラス{label}"
            suffix = 2
            chosen_name = f"{base}_{suffix}"
            while chosen_name in used_names:
                suffix += 1
                chosen_name = f"{base}_{suffix}"

        names[label] = chosen_name
        used_names.add(chosen_name)

    return names


def find_representative_titles(
    doc_vectors: np.ndarray,
    labels: np.ndarray,
    titles: list[str],
    top_n: int = DEFAULT_TOP_N_REPRESENTATIVES,
) -> dict[int, list[str]]:
    """クラスタごとに、重心とのコサイン類似度が最も高い（=最も近い）
    記事タイトルを最大top_n件返す。
    """
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels))

    representatives: dict[int, list[str]] = {}
    for label in unique_labels:
        mask = labels == label
        member_vectors = doc_vectors[mask]
        member_titles = [title for title, included in zip(titles, mask) if included]

        centroid = member_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            similarities = np.zeros(len(member_vectors))
        else:
            # 文書ベクトルは既にL2正規化済み（embed_texts）なので、
            # コサイン類似度は dot(doc, centroid) / ||centroid|| に簡略化できる
            similarities = (member_vectors @ centroid) / centroid_norm

        ranked = np.argsort(similarities)[::-1][:top_n]
        representatives[label] = [member_titles[i] for i in ranked]

    return representatives
