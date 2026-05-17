import logging
from typing import List

from elasticsearch import Elasticsearch
from langchain_core.documents import Document

from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_es_client() -> Elasticsearch:
    settings = get_settings()
    return Elasticsearch(settings.elasticsearch_url, request_timeout=5)


def index_documents(chunks: List[Document]) -> None:
    """Index document chunks into Elasticsearch for BM25 retrieval."""
    try:
        settings = get_settings()
        es = _get_es_client()

        if not es.indices.exists(index=settings.elasticsearch_index):
            es.indices.create(
                index=settings.elasticsearch_index,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text", "analyzer": "english"},
                            "source": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "chunk_index": {"type": "integer"},
                        }
                    }
                },
            )

        operations = []
        for i, chunk in enumerate(chunks):
            operations.append({"index": {"_index": settings.elasticsearch_index}})
            operations.append(
                {
                    "content": chunk.page_content,
                    "source": chunk.metadata.get("source", "unknown"),
                    "page": chunk.metadata.get("page", 0),
                    "chunk_index": chunk.metadata.get("chunk_index", i),
                }
            )

        if operations:
            es.bulk(body=operations, refresh=True)
    except Exception:
        logger.error("Failed to index documents into Elasticsearch", exc_info=True)
        raise


def bm25_search(query: str, top_k: int) -> List[Document]:
    """Run a BM25 query against Elasticsearch and return matching Documents."""
    try:
        settings = get_settings()
        es = _get_es_client()

        if not es.indices.exists(index=settings.elasticsearch_index):
            return []

        response = es.search(
            index=settings.elasticsearch_index,
            body={
                "query": {"match": {"content": {"query": query, "operator": "or"}}},
                "size": top_k,
            },
        )

        docs = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            docs.append(
                Document(
                    page_content=src["content"],
                    metadata={
                        "source": src.get("source", "unknown"),
                        "page": src.get("page", 0),
                        "chunk_index": src.get("chunk_index", 0),
                        "bm25_score": hit["_score"],
                    },
                )
            )
        return docs
    except Exception:
        logger.warning("BM25 search failed, returning empty results", exc_info=True)
        return []
