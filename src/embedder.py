import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode texts into L2-normalized dense vectors for clustering."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    prefixed = [f"query: {text}" for text in texts]
    return _get_model().encode(
        prefixed, normalize_embeddings=True, convert_to_numpy=True
    )
