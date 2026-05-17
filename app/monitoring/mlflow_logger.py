import json
from typing import List

import mlflow
from langchain_core.documents import Document

from app.config import get_settings

_initialized = False


def _ensure_experiment() -> None:
    global _initialized
    if _initialized:
        return

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    existing = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
    if existing is None:
        mlflow.create_experiment(settings.mlflow_experiment_name)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    _initialized = True


def log_query_run(
    query: str,
    answer: str,
    retrieved_docs: List[Document],
    retrieval_latency_ms: float,
    generation_latency_ms: float,
) -> None:
    """Log a single RAG query/response run to MLflow."""
    try:
        _ensure_experiment()

        rerank_scores = [
            doc.metadata.get("rerank_score", 0.0) for doc in retrieved_docs
        ]
        source_files = list({doc.metadata.get("source", "unknown") for doc in retrieved_docs})

        with mlflow.start_run():
            mlflow.log_param("query", query[:250])
            mlflow.log_param("num_docs_retrieved", len(retrieved_docs))
            mlflow.log_param("source_files", json.dumps(source_files))

            mlflow.log_metric("retrieval_latency_ms", retrieval_latency_ms)
            mlflow.log_metric("generation_latency_ms", generation_latency_ms)
            mlflow.log_metric("total_latency_ms", retrieval_latency_ms + generation_latency_ms)

            if rerank_scores:
                mlflow.log_metric("rerank_score_top1", rerank_scores[0])
                mlflow.log_metric("rerank_score_mean", sum(rerank_scores) / len(rerank_scores))

            # Log full answer as artifact
            mlflow.log_text(answer, "answer.txt")
            mlflow.log_text(
                json.dumps(
                    [
                        {
                            "content": doc.page_content[:300],
                            "source": doc.metadata.get("source"),
                            "rerank_score": doc.metadata.get("rerank_score"),
                        }
                        for doc in retrieved_docs
                    ],
                    indent=2,
                ),
                "retrieved_docs.json",
            )
    except Exception:
        # Monitoring failures must never break the main query path
        pass
