"""Answers carry a response_id, and /feedback records a rating without any question text."""
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app

client = TestClient(app)

_DOCS = [Document(page_content="FAFSA code 002501", metadata={"source": "semo.edu/fafsa", "rerank_score": 5.0})]


@patch("app.api.routes.query.retrieve_context", return_value=_DOCS)
@patch("app.api.routes.query.generate_answer", return_value="Use code 002501.")
@patch("app.api.routes.query.log_query_run")
def test_query_returns_response_id_and_logs_it(mock_log, _gen, _ret):
    data = client.post("/query", json={"question": "What is the SEMO FAFSA code?"}).json()
    assert len(data["response_id"]) == 32
    assert mock_log.call_args.args[-1] == data["response_id"]


@patch("app.api.routes.feedback.log_feedback")
def test_feedback_logs_rating(mock_log):
    rid = "a" * 32
    resp = client.post("/feedback", json={"response_id": rid, "rating": "down"})
    assert resp.status_code == 200
    mock_log.assert_called_once_with(rid, "down")


@patch("app.api.routes.feedback.log_feedback")
def test_feedback_rejects_bad_input(mock_log):
    assert client.post("/feedback", json={"response_id": "x", "rating": "up"}).status_code == 422
    assert client.post("/feedback", json={"response_id": "a" * 32, "rating": "meh"}).status_code == 422
    mock_log.assert_not_called()
