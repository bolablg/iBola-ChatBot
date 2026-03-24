"""
Pydantic-settings based configuration for the iBola chatbot.

Loads from environment variables and .env files with type validation.
This is the modern replacement for the flat config.py module.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMSettings(BaseSettings):
    """LLM / Gemini configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    model_name: str = "gemini-2.5-flash"
    guardrail_temperature: float = 0.0
    grading_temperature: float = 0.0
    rewrite_temperature: float = 0.3
    generation_temperature: float = 0.7
    guardrail_threshold: int = 60
    max_retrieval_attempts: int = 2


class SearchSettings(BaseSettings):
    """Hybrid search configuration."""

    model_config = SettingsConfigDict(env_prefix="SEARCH_")

    use_hybrid: bool = True
    use_reranker: bool = True
    rrf_rank_constant: int = 60
    vector_top_k: int = 8
    vector_fetch_k: int = 24
    bm25_top_k: int = 8
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CacheSettings(BaseSettings):
    """Cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    response_ttl: int = 1800  # 30 minutes
    session_ttl: int = 3600  # 1 hour
    language_ttl: int = 7200  # 2 hours
    redis_url: Optional[str] = None


class TracingSettings(BaseSettings):
    """Langfuse tracing configuration."""

    model_config = SettingsConfigDict(env_prefix="LANGFUSE_")

    enabled: bool = False
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: str = "https://cloud.langfuse.com"


class AppSettings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API keys ---
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # --- GCP ---
    gcp_project_id: str = Field(default="", alias="GCP_PROJECT_ID")
    gcp_sa_credentials_path: str = Field(
        default=str(_PROJECT_ROOT / "_conf" / "ibola_agent_sa.json"),
        alias="GCP_SA_CREDENTIALS_PATH",
    )

    # --- Paths ---
    db_path: str = Field(default=str(_PROJECT_ROOT / "chroma_db"), alias="DB_PATH")
    data_path: str = Field(default=str(_PROJECT_ROOT / "data"), alias="DATA_PATH")

    # --- Integrations ---
    gchat_webhook_url: Optional[str] = Field(default=None, alias="GCHAT_WEBHOOK_URL")
    redirect_log_sheet_id: Optional[str] = Field(
        default=None, alias="REDIRECT_LOG_SHEET_ID"
    )
    google_oauth_credentials_path: Optional[str] = Field(
        default=None, alias="GOOGLE_OAUTH_CREDENTIALS_PATH"
    )

    # --- Server ---
    host: str = Field(
        default="0.0.0.0",  # nosec B104 - required for Docker/Cloud Run
        alias="HOST",
    )
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # --- Security ---
    allowed_origin_regex: str = Field(
        default=r"https://(.+\.)?bolablg\.com",
        alias="ALLOWED_ORIGIN_REGEX",
    )

    # --- Session ---
    session_timeout_minutes: int = Field(default=30, alias="SESSION_TIMEOUT_MINUTES")
    max_redirect_count: int = Field(default=3, alias="MAX_REDIRECT_COUNT")

    # --- Languages ---
    supported_languages: List[str] = Field(
        default=["en", "fr"],
    )

    # --- Contact ---
    contact_email: str = "hello@bolablg.com"
    calendar_booking_url: str = (
        "https://calendar.google.com/calendar/appointments/schedules/"
        "AcZssZ3YeidR5Og4YSGZIlxUIlDAf0AiRA6N8-MAzr-Sy55BtbKhBLXkfa8M_P_92eokXRnayLVlEXiW?gv=true"
    )
    linkedin_url: str = "https://linkedin.com/in/bolablg"

    # --- Sub-settings ---
    llm: LLMSettings = Field(default_factory=LLMSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)

    @field_validator("gemini_api_key")
    @classmethod
    def validate_gemini_key(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "GEMINI_API_KEY is required. Set it in .env or as an environment variable."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings singleton."""
    return AppSettings()
