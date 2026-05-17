import logging
from fastapi import APIRouter, HTTPException
from app.api.schemas import IngestUrlRequest, IngestUrlResponse
from app.config import get_settings
from app.ingestion.scraper import scrape_url, scrape_recursive
from app.ingestion.splitter import split_documents
from app.rag.vector_store import add_documents_to_store
from app.rag.bm25_retriever import index_documents

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest/url", response_model=IngestUrlResponse, summary="Scrape and index URLs from semo.edu")
async def ingest_url(request: IngestUrlRequest) -> IngestUrlResponse:
    settings = get_settings()
    all_docs = []
    all_errors = []

    for url in request.urls:
        try:
            if request.recursive:
                docs, errors = scrape_recursive(
                    url,
                    allowed_domain=settings.allowed_scrape_domain,
                    max_depth=settings.scrape_max_depth,
                    max_pages=settings.scrape_max_pages,
                )
            else:
                docs = scrape_url(url, allowed_domain=settings.allowed_scrape_domain)
                errors = []
            all_docs.extend(docs)
            all_errors.extend(errors)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            all_errors.append({"url": url, "error": str(exc)})

    if not all_docs and all_errors:
        raise HTTPException(status_code=502, detail={"message": "All URLs failed to scrape", "errors": all_errors})

    chunks = split_documents(all_docs)
    add_documents_to_store(chunks)
    try:
        index_documents(chunks)
    except Exception:
        logger.warning("BM25 indexing failed; chunks are in vector store only")

    return IngestUrlResponse(
        urls_submitted=len(request.urls),
        pages_scraped=len(all_docs),
        chunks_indexed=len(chunks),
        errors=all_errors,
    )
