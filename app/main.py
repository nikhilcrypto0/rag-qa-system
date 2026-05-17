import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import ingest, query, ingest_url
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import limiter, RateLimitExceeded, _rate_limit_exceeded_handler
from app.rag.embeddings import get_embeddings
from app.rag.reranker import _get_cross_encoder
from app.rag.vector_store import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load models and vector store at startup to avoid cold-start on first request
    get_embeddings()
    _get_cross_encoder()
    get_vector_store()
    yield


app = FastAPI(
    title="RAG Q&A System",
    description=(
        "Production-grade domain-specific Question Answering via hybrid retrieval "
        "(BM25 + dense embeddings), cross-encoder re-ranking, and Claude-powered generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(ingest_url.router, tags=["Ingestion"])
app.include_router(query.router, tags=["Query"])


@app.get("/health", tags=["Health"])
def health() -> dict:
    return {"status": "ok"}


frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/ui", StaticFiles(directory=frontend_dist, html=True), name="frontend")
