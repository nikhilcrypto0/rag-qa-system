# RAG QA System

A production-ready Retrieval-Augmented Generation question-answering system. Upload documents or URLs, ask questions, get grounded answers with source citations.

## Features

- **Hybrid retrieval** — combines dense (FAISS/Pinecone) and sparse (BM25) search with a re-ranker
- **Multiple vector stores** — FAISS (local), Pinecone, and Elasticsearch backends
- **Document ingestion** — PDF, DOCX, plain text, and web URL scraping
- **Claude-powered generation** — answers grounded in retrieved context
- **MLflow tracking** — logs retrieval metrics and generation quality per query
- **Auth + rate limiting** — API key middleware and per-client rate limits
- **Docker** — single `docker-compose up` to run the full stack

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain |
| Embeddings | Sentence Transformers |
| Vector stores | FAISS, Pinecone, Elasticsearch |
| LLM | Claude (Anthropic API) |
| Re-ranking | Cross-encoder |
| API | FastAPI |
| Monitoring | MLflow |
| Frontend | React + Vite |

## Project Structure

```
app/
├── api/routes/    # Ingest, query, and URL endpoints
├── rag/           # Pipeline, retrieval, re-ranking, generation
├── ingestion/     # Document loaders and text splitters
├── middleware/    # Auth and rate limiting
└── monitoring/    # MLflow logging

docker/
├── Dockerfile
└── docker-compose.yml

frontend/          # React query interface
```

## Quickstart

```bash
# Docker (recommended)
docker-compose -f docker/docker-compose.yml up

# Or locally
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY + vector store credentials
uvicorn app.main:app --reload
```

## API

```
POST /ingest          # Upload a document file
POST /ingest/url      # Ingest a web URL
POST /query           # Ask a question
```

## Environment Variables

```
ANTHROPIC_API_KEY=
PINECONE_API_KEY=         # optional
ELASTICSEARCH_URL=        # optional
```
