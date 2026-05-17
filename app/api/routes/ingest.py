import os
import tempfile
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.api.schemas import IngestResponse
from app.ingestion.loaders import load_document
from app.ingestion.splitter import split_documents
from app.rag.bm25_retriever import index_documents
from app.rag.vector_store import add_documents_to_store

router = APIRouter()

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc"}


@router.post("/ingest", response_model=IngestResponse, summary="Upload and index documents")
async def ingest_documents(files: List[UploadFile] = File(...)) -> IngestResponse:
    """
    Accept one or more documents (PDF, TXT, DOCX), chunk them, and index
    into both the vector store (FAISS/Pinecone) and Elasticsearch (BM25).
    """
    total_chunks = 0

    for upload in files:
        filename = upload.filename or "upload"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}",
            )

        # Write to a temp file so loaders can use the file path
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await upload.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            docs = load_document(tmp_path)
            # Preserve original filename in metadata
            for doc in docs:
                doc.metadata["source"] = filename

            chunks = split_documents(docs)
            add_documents_to_store(chunks)
            index_documents(chunks)
            total_chunks += len(chunks)
        finally:
            os.unlink(tmp_path)

    return IngestResponse(
        message=f"Successfully indexed {len(files)} file(s).",
        files_processed=len(files),
        chunks_indexed=total_chunks,
    )
