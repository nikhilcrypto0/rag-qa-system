import os
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.config import get_settings
from app.ingestion.splitter import chunk_id
from app.rag.embeddings import get_embeddings

_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the active vector store (FAISS or Pinecone), initializing if needed."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        if settings.vector_store == "pinecone":
            _vector_store = _init_pinecone()
        else:
            _vector_store = _init_faiss()
    return _vector_store


def _init_faiss() -> VectorStore:
    from langchain_community.vectorstores import FAISS

    settings = get_settings()
    index_path = settings.faiss_index_path

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        return FAISS.load_local(index_path, get_embeddings(), allow_dangerous_deserialization=True)

    # Create empty FAISS store with a placeholder doc so the index is valid
    placeholder = Document(page_content="placeholder", metadata={"source": "__init__"})
    store = FAISS.from_documents([placeholder], get_embeddings())
    os.makedirs(index_path, exist_ok=True)
    store.save_local(index_path)
    return store


def _init_pinecone() -> VectorStore:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing = [i.name for i in pc.list_indexes()]
    if settings.pinecone_index not in existing:
        pc.create_index(
            name=settings.pinecone_index,
            dimension=384,  # all-MiniLM-L6-v2 output dim
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    return PineconeVectorStore(
        index_name=settings.pinecone_index,
        embedding=get_embeddings(),
        pinecone_api_key=settings.pinecone_api_key,
    )


def add_documents_to_store(chunks: List[Document]) -> None:
    """Index chunks into the active vector store and persist if FAISS. Already-indexed chunks are skipped."""
    settings = get_settings()
    store = get_vector_store()

    ids = [chunk_id(c) for c in chunks]
    if settings.vector_store == "faiss":
        from langchain_community.vectorstores import FAISS
        assert isinstance(store, FAISS)
        existing = set(store.index_to_docstore_id.values())
        new = {i: c for i, c in zip(ids, chunks) if i not in existing}  # dict also drops in-batch repeats
        if new:
            store.add_documents(list(new.values()), ids=list(new.keys()))
        store.save_local(settings.faiss_index_path)
    else:
        store.add_documents(chunks, ids=ids)  # Pinecone upserts by ID
