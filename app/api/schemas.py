from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_indexed: int


class SourceDocument(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    rerank_score: Optional[float] = None
    excerpt: str = Field(description="First 300 chars of the chunk")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    response_id: str = Field(default="", description="Pass to /feedback; empty for off-topic replies")
    answer: str
    sources: List[SourceDocument]
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    retrieval_mode: str = Field(default="hybrid", description='"hybrid" or "dense_only" when Elasticsearch is down')


class FeedbackRequest(BaseModel):
    response_id: str = Field(..., pattern=r"^[0-9a-f]{32}$")
    rating: Literal["up", "down"]


class IngestUrlRequest(BaseModel):
    urls: List[str] = Field(..., min_length=1, max_length=10, description="List of URLs to scrape and index")
    recursive: bool = Field(default=True, description="Follow internal links (depth 2)")


class IngestUrlResponse(BaseModel):
    urls_submitted: int
    pages_scraped: int
    chunks_indexed: int
    errors: List[dict] = []
