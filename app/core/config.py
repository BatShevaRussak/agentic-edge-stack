"""Application configuration via Pydantic Settings (loaded from .env)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings."""

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    request_timeout: int = 120

    # RAG
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50
    rag_top_k: int = 3
    rag_score_threshold: float = 0.5
    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")

    # API (Part 4)
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: list[str] = ["*"]
    sse_keepalive_seconds: int = 15

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
