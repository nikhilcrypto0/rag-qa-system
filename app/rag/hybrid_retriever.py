from typing import List, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import get_settings
from app.rag.bm25_retriever import bm25_search
from app.rag.reranker import rerank
from app.rag.vector_store import get_vector_store


def _deduplicate(docs: List[Document]) -> List[Document]:
    """Remove duplicate documents by content hash."""
    seen: set = set()
    unique: List[Document] = []
    for doc in docs:
        key = hash(doc.page_content[:200])
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


class HybridRetriever(BaseRetriever):
    """Combines BM25 sparse retrieval + dense vector retrieval, then cross-encoder re-ranks."""

    top_k: int = 10
    top_n: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. BM25 sparse retrieval
        try:
            bm25_docs = bm25_search(query, top_k=self.top_k)
        except Exception:
            bm25_docs = []

        # 2. Dense vector retrieval
        store = get_vector_store()
        dense_retriever: VectorStoreRetriever = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )
        dense_docs = dense_retriever.invoke(query)

        # 3. Merge and deduplicate
        candidates = _deduplicate(bm25_docs + dense_docs)

        # 4. Cross-encoder re-ranking
        ranked: List[Tuple[Document, float]] = rerank(query, candidates, top_n=self.top_n)

        # 5. Attach rerank score to metadata and return
        results = []
        for doc, score in ranked:
            doc.metadata["rerank_score"] = round(float(score), 4)
            results.append(doc)

        return results


def build_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        top_k=settings.top_k_retrieval,
        top_n=settings.top_k_rerank,
    )
