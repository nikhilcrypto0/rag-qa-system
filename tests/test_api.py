"""Tests for FastAPI endpoints (mocked RAG pipeline)."""
import io
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app

client = TestClient(app)

_MOCK_DOCS = [
    Document(
        page_content="Transformers use self-attention mechanisms.",
        metadata={"source": "transformers.txt", "chunk_index": 1, "rerank_score": 0.92},
    )
]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("app.api.routes.query.retrieve_context", return_value=_MOCK_DOCS)
@patch("app.api.routes.query.generate_answer", return_value="Transformers use self-attention. [Source: transformers.txt, chunk 1]")
@patch("app.api.routes.query.log_query_run")
def test_query_endpoint(mock_log, mock_gen, mock_retrieve):
    response = client.post("/query", json={"question": "What is a Transformer?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "transformers.txt"
    assert "retrieval_latency_ms" in data
    assert "generation_latency_ms" in data


@patch("app.api.routes.query.retrieve_context", return_value=_MOCK_DOCS)
@patch("app.api.routes.query.generate_answer", return_value="Some answer.")
@patch("app.api.routes.query.log_query_run")
def test_query_returns_latency_fields(mock_log, mock_gen, mock_retrieve):
    response = client.post("/query", json={"question": "Test question", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["total_latency_ms"] >= 0


def test_query_empty_question_rejected():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


@patch("app.api.routes.ingest.load_document", return_value=_MOCK_DOCS)
@patch("app.api.routes.ingest.split_documents", return_value=_MOCK_DOCS)
@patch("app.api.routes.ingest.add_documents_to_store")
@patch("app.api.routes.ingest.index_documents")
def test_ingest_txt_file(mock_es_idx, mock_vec_idx, mock_split, mock_load):
    fake_file = io.BytesIO(b"This is a test document about AI.")
    response = client.post(
        "/ingest",
        files=[("files", ("test.txt", fake_file, "text/plain"))],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["files_processed"] == 1
    assert data["chunks_indexed"] >= 0


def test_ingest_unsupported_extension():
    fake_file = io.BytesIO(b"data")
    response = client.post(
        "/ingest",
        files=[("files", ("data.csv", fake_file, "text/csv"))],
    )
    assert response.status_code == 415
