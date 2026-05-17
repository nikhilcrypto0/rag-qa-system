"""Tests for document loading and splitting."""
import os
import tempfile

import pytest
from langchain_core.documents import Document

from app.ingestion.loaders import load_document
from app.ingestion.splitter import split_documents


def _write_temp_txt(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    f.write(content)
    f.flush()
    f.close()
    return f.name


def test_load_txt_document():
    path = _write_temp_txt("Hello world. This is a test document.")
    try:
        docs = load_document(path)
        assert len(docs) >= 1
        assert "Hello world" in docs[0].page_content
    finally:
        os.unlink(path)


def test_load_document_sets_source_metadata():
    path = _write_temp_txt("Sample content.")
    try:
        docs = load_document(path)
        assert all("source" in doc.metadata for doc in docs)
    finally:
        os.unlink(path)


def test_load_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document("some_file.xyz")


def test_split_documents_creates_chunks():
    docs = [Document(page_content="word " * 300, metadata={"source": "test.txt"})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_split_preserves_source_metadata():
    docs = [Document(page_content="word " * 200, metadata={"source": "myfile.txt"})]
    chunks = split_documents(docs)
    for chunk in chunks:
        assert chunk.metadata.get("source") == "myfile.txt"


def test_split_assigns_chunk_index():
    docs = [Document(page_content="word " * 300, metadata={"source": "test.txt"})]
    chunks = split_documents(docs, chunk_size=100, chunk_overlap=0)
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata
