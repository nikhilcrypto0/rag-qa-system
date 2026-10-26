"""Tests for privacy-safe logging, polite scraping, and visible ES fallback."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.ingestion import scraper


@patch("app.rag.hybrid_retriever.rerank")
@patch("app.rag.hybrid_retriever.get_vector_store")
@patch("app.rag.hybrid_retriever.bm25_search", side_effect=ConnectionError("es down"))
def test_es_down_marks_results_dense_only(_bm25, mock_store, mock_rerank):
    doc = Document(page_content="Financial aid info", metadata={"source": "semo.edu/aid"})
    mock_store.return_value.as_retriever.return_value.invoke.return_value = [doc]
    mock_rerank.return_value = [(doc, 0.9)]

    from app.rag.hybrid_retriever import HybridRetriever
    results = HybridRetriever(top_k=5, top_n=3).invoke("financial aid")

    assert results[0].metadata["retrieval_mode"] == "dense_only"


def test_scraper_refuses_robots_disallowed_url():
    scraper._robots_cache.clear()
    robots = MagicMock(ok=True, text="User-agent: *\nDisallow: /private")
    with patch("app.ingestion.scraper.requests.get", return_value=robots) as mock_get, \
         patch("app.ingestion.scraper.time.sleep"):
        with pytest.raises(PermissionError):
            scraper.scrape_url("https://semo.edu/private/page", allowed_domain="semo.edu")
    assert mock_get.call_count == 1  # only robots.txt was fetched, never the page


def test_scraper_waits_before_each_request():
    scraper._robots_cache.clear()
    robots = MagicMock(ok=True, text="User-agent: *\nAllow: /")
    page = MagicMock(text="<html><main>Hello SEMO</main></html>")
    with patch("app.ingestion.scraper.requests.get", side_effect=[robots, page]), \
         patch("app.ingestion.scraper.time.sleep") as mock_sleep:
        docs = scraper.scrape_url("https://semo.edu/about", allowed_domain="semo.edu")
    assert docs and mock_sleep.called


@patch("app.monitoring.mlflow_logger.mlflow")
@patch("app.monitoring.mlflow_logger._ensure_experiment")
def test_mlflow_never_logs_raw_question(_ensure, mock_mlflow):
    from app.monitoring.mlflow_logger import log_query_run
    question = "My student ID is 1234567, am I on probation?"
    log_query_run(question, "answer", [], 10.0, 20.0)

    logged = [str(c) for c in mock_mlflow.log_param.call_args_list]
    assert not any("1234567" in entry for entry in logged)
    assert any("query_hash" in entry for entry in logged)
