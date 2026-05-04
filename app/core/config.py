"""Application configuration using Pydantic Settings.

All configuration is loaded from environment variables (or .env file).
This is the SINGLE source of truth for application settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings."""

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"

    request_timeout: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
