"""
Rebuild a deduplicated copy of the index from an existing FAISS store.

Writes a NEW FAISS folder and a NEW Elasticsearch index (ELASTICSEARCH_INDEX env var);
the source index is only read, so the old one stays as a backup until you switch config.

Usage:
  ELASTICSEARCH_INDEX=semo-documents-v2 PYTHONPATH=. .venv/bin/python scripts/rebuild_index.py \
      data/faiss_index data/faiss_index_v2
"""
import sys

from langchain_community.vectorstores import FAISS

from app.ingestion.splitter import chunk_id
from app.rag.bm25_retriever import index_documents
from app.rag.embeddings import get_embeddings


def main(source_path: str, out_path: str) -> None:
    source = FAISS.load_local(source_path, get_embeddings(), allow_dangerous_deserialization=True)
    docs = [d for d in source.docstore._dict.values() if d.metadata.get("source") != "__init__"]

    unique = {chunk_id(d): d for d in docs}
    print(f"{len(docs)} chunks in source, {len(unique)} unique")

    store = FAISS.from_documents(list(unique.values()), get_embeddings(), ids=list(unique.keys()))
    store.save_local(out_path)
    print(f"FAISS written to {out_path}")

    index_documents(list(unique.values()))
    print("Elasticsearch index written")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
