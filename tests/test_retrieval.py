"""Tests for retrieval components (mocked external services)."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.reranker import rerank


def _make_docs(n: int) -> list:
    return [
        Document(page_content=f"Document number {i} about machine learning.", metadata={"source": f"doc{i}.txt"})
        for i in range(n)
    ]


@patch("app.rag.reranker._get_cross_encoder")
def test_rerank_returns_top_n(mock_encoder_factory):
    mock_encoder = MagicMock()
    import numpy as np
    mock_encoder.predict.return_value = np.array([0.9, 0.3, 0.7, 0.1, 0.5])
    mock_encoder_factory.return_value = mock_encoder

    docs = _make_docs(5)
    results = rerank("What is machine learning?", docs, top_n=3)

    assert len(results) == 3
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


@patch("app.rag.reranker._get_cross_encoder")
def test_rerank_empty_candidates(mock_encoder_factory):
    results = rerank("query", [], top_n=5)
    assert results == []


@patch("app.rag.reranker._get_cross_encoder")
def test_rerank_fewer_candidates_than_top_n(mock_encoder_factory):
    mock_encoder = MagicMock()
    import numpy as np
    mock_encoder.predict.return_value = np.array([0.8, 0.2])
    mock_encoder_factory.return_value = mock_encoder

    docs = _make_docs(2)
    results = rerank("query", docs, top_n=10)
    assert len(results) == 2


@patch("app.rag.bm25_retriever._get_es_client")
def test_bm25_search_no_index_returns_empty(mock_es_client):
    mock_es = MagicMock()
    mock_es.indices.exists.return_value = False
    mock_es_client.return_value = mock_es

    from app.rag.bm25_retriever import bm25_search
    results = bm25_search("test query", top_k=5)
    assert results == []
