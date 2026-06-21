"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for SafeAgent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="safeagent", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")

    faiss_index_path: str = Field(default="faiss_index/", alias="FAISS_INDEX_PATH")
    safety_log_path: str = Field(default="safety_log.jsonl", alias="SAFETY_LOG_PATH")

    safety_pass_threshold: float = Field(default=0.8, alias="SAFETY_PASS_THRESHOLD")
    safety_warn_threshold: float = Field(default=0.5, alias="SAFETY_WARN_THRESHOLD")
    max_safety_retries: int = Field(default=2, alias="MAX_SAFETY_RETRIES")

    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    app_version: str = "0.1.0"
    port: int = Field(default=8080, alias="PORT")

    min_faiss_hits: int = 2
    faiss_top_k: int = 3
    web_search_max_results: int = 5
    safety_log_limit: int = 50


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
