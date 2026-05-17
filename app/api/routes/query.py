import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.api.schemas import QueryRequest, QueryResponse, SourceDocument
from app.config import get_settings
from app.monitoring.mlflow_logger import log_query_run
from app.rag.generator import generate_answer, stream_answer
from app.rag.guardrails import is_university_topic, OFF_TOPIC_RESPONSE
from app.rag.pipeline import retrieve_context

router = APIRouter()


def _doc_to_source(doc) -> SourceDocument:
    return SourceDocument(
        source=doc.metadata.get("source", "unknown"),
        page=doc.metadata.get("page"),
        chunk_index=doc.metadata.get("chunk_index"),
        rerank_score=doc.metadata.get("rerank_score"),
        excerpt=doc.page_content[:300],
    )


@router.post("/query", response_model=QueryResponse, summary="Ask a question (non-streaming)")
async def query(request: QueryRequest, background_tasks: BackgroundTasks) -> QueryResponse:
    """Retrieve relevant context, generate a grounded answer, and return with source citations."""

    # Guardrail pre-filter
    settings = get_settings()
    if settings.enable_domain_guardrails and not is_university_topic(request.question):
        return QueryResponse(
            answer=OFF_TOPIC_RESPONSE,
            sources=[],
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            total_latency_ms=0.0,
        )

    # Retrieval
    t0 = time.perf_counter()
    docs = retrieve_context(request.question)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Generation
    t1 = time.perf_counter()
    answer = generate_answer(request.question, docs)
    generation_ms = (time.perf_counter() - t1) * 1000

    # Log to MLflow (non-blocking)
    background_tasks.add_task(log_query_run, request.question, answer, docs, retrieval_ms, generation_ms)

    return QueryResponse(
        answer=answer,
        sources=[_doc_to_source(d) for d in docs],
        retrieval_latency_ms=round(retrieval_ms, 2),
        generation_latency_ms=round(generation_ms, 2),
        total_latency_ms=round(retrieval_ms + generation_ms, 2),
    )


@router.post("/query/stream", summary="Ask a question (streaming SSE)")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Stream the answer token-by-token via Server-Sent Events."""

    # Guardrail pre-filter
    settings = get_settings()
    if settings.enable_domain_guardrails and not is_university_topic(request.question):
        async def off_topic_stream():
            yield f"data: {OFF_TOPIC_RESPONSE}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(off_topic_stream(), media_type="text/event-stream")

    docs = retrieve_context(request.question)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for token in stream_answer(request.question, docs):
            yield f"data: {json.dumps(token)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
