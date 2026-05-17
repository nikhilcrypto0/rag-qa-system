import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader


def load_document(file_path: str) -> List[Document]:
    """Load a single document based on its file extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext in (".docx", ".doc"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()

    # Ensure source metadata is set
    for doc in docs:
        doc.metadata.setdefault("source", os.path.basename(file_path))

    return docs


def load_documents_from_dir(directory: str) -> List[Document]:
    """Load all supported documents from a directory."""
    supported = {".pdf", ".txt", ".md", ".docx", ".doc"}
    all_docs: List[Document] = []

    for path in Path(directory).iterdir():
        if path.suffix.lower() in supported:
            all_docs.extend(load_document(str(path)))

    return all_docs
