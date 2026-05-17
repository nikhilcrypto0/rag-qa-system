from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.config import get_settings


@lru_cache(maxsize=1)
def _get_cross_encoder() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.cross_encoder_model)


def rerank(query: str, candidates: List[Document], top_n: int) -> List[Tuple[Document, float]]:
    """Score (query, doc) pairs with a cross-encoder and return top_n ranked results."""
    if not candidates:
        return []

    encoder = _get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in candidates]
    scores: List[float] = encoder.predict(pairs).tolist()

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
