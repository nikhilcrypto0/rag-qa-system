from typing import AsyncGenerator, List

import anthropic
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.guardrails import get_domain_system_prompt


def _build_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk_index", i)
        parts.append(f"[Passage {i} | Source: {source}, chunk {chunk}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_user_message(query: str, docs: List[Document]) -> str:
    context = _build_context(docs)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_answer(query: str, docs: List[Document]) -> str:
    """Non-streaming answer generation via Claude API."""
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    message = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        system=get_domain_system_prompt(),
        messages=[{"role": "user", "content": _build_user_message(query, docs)}],
    )
    return message.content[0].text


async def stream_answer(query: str, docs: List[Document]) -> AsyncGenerator[str, None]:
    """Streaming answer generation via Claude API using SSE."""
    settings = get_settings()
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async with client.messages.stream(
        model=settings.claude_model,
        max_tokens=1024,
        system=get_domain_system_prompt(),
        messages=[{"role": "user", "content": _build_user_message(query, docs)}],
    ) as stream:
        async for text_chunk in stream.text_stream:
            yield text_chunk
