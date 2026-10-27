"""Re-ingesting the same chunks must not create duplicates in FAISS or Elasticsearch."""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.ingestion.splitter import chunk_id


def _chunk(text: str, source: str = "https://semo.edu/a") -> Document:
    return Document(page_content=text, metadata={"source": source})


def test_chunk_id_depends_on_source_and_text():
    assert chunk_id(_chunk("x")) == chunk_id(_chunk("x"))
    assert chunk_id(_chunk("x")) != chunk_id(_chunk("y"))
    assert chunk_id(_chunk("x", "https://semo.edu/a")) != chunk_id(_chunk("x", "https://semo.edu/b"))


def test_faiss_skips_chunks_already_indexed():
    from langchain_community.vectorstores import FAISS
    from app.rag import vector_store

    a, b = _chunk("aid"), _chunk("tuition")
    store = MagicMock(spec=FAISS)
    store.index_to_docstore_id = {0: chunk_id(a)}  # "a" is already in the index
    settings = MagicMock(vector_store="faiss", faiss_index_path="/tmp/x")
    with patch.object(vector_store, "get_vector_store", return_value=store), \
         patch.object(vector_store, "get_settings", return_value=settings):
        vector_store.add_documents_to_store([a, b, b])

    store.add_documents.assert_called_once_with([b], ids=[chunk_id(b)])


def test_elasticsearch_uses_stable_ids():
    from app.rag import bm25_retriever

    es = MagicMock()
    es.indices.exists.return_value = True
    with patch.object(bm25_retriever, "_get_es_client", return_value=es):
        bm25_retriever.index_documents([_chunk("aid")])

    ops = es.bulk.call_args.kwargs["body"]
    assert ops[0]["index"]["_id"] == chunk_id(_chunk("aid"))
