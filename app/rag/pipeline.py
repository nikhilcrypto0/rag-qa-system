from typing import List
from langchain_core.documents import Document
from app.rag.hybrid_retriever import build_hybrid_retriever


def retrieve_context(query: str) -> List[Document]:
    """Retrieve relevant documents for a query without running generation."""
    retriever = build_hybrid_retriever()
    return retriever.invoke(query)
