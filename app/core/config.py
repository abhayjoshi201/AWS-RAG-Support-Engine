"""Centralised configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All env vars consumed by the RAG application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── AWS / Bedrock ───────────────────────────────────
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    bedrock_llm_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"

    # ── OpenSearch ──────────────────────────────────────
    opensearch_host: str = "https://localhost"
    opensearch_port: int = 9200
    opensearch_username: str = "admin"
    opensearch_password: str = ""
    opensearch_index: str = "rag-knowledge-base"
    opensearch_use_ssl: bool = True
    opensearch_verify_certs: bool = False

    # ── Zendesk ─────────────────────────────────────────
    zendesk_subdomain: str = ""
    zendesk_email: str = ""
    zendesk_api_token: str = ""

    # ── Antigravity Webhook ─────────────────────────────
    antigravity_webhook_secret: str = ""

    # ── App tunables ────────────────────────────────────
    log_level: str = "INFO"
    vector_dimension: int = 1024
    vector_top_k: int = 5
    demo_mode: bool = False

    @property
    def zendesk_base_url(self) -> str:
        return f"https://{self.zendesk_subdomain}.zendesk.com/api/v2"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
