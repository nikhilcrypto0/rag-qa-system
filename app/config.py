from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""

    # Vector store
    vector_store: str = "faiss"  # "faiss" | "pinecone"
    pinecone_api_key: str = ""
    pinecone_index: str = "rag-qa-index"
    pinecone_environment: str = ""

    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "rag-documents"

    # Models
    claude_model: str = "claude-haiku-4-5"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval
    top_k_retrieval: int = 6
    top_k_rerank: int = 3

    # FAISS
    faiss_index_path: str = "data/faiss_index"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "rag-qa-system"

    # Auth
    auth_mode: str = "none"  # "none" | "api_key" | "email_domain"
    api_keys: str = ""
    allowed_email_domain: str = "semo.edu"

    # Rate limiting
    rate_limit_query: str = "20/minute"
    rate_limit_ingest: str = "5/minute"

    # Guardrails
    enable_domain_guardrails: bool = True

    # Scraper
    allowed_scrape_domain: str = "semo.edu"
    scrape_max_depth: int = 2
    scrape_max_pages: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
