from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with explicit local-development and durability boundaries."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="forbid", populate_by_name=True)

    app_name: str = "PrivateAIStack"
    environment: str = Field(default="development", validation_alias="PRIVATE_AI_STACK_ENVIRONMENT")
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_key: str | None = None

    database_url: str = "postgresql://private_ai_stack:private_ai_stack@postgres:5432/private_ai_stack"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_timeout_seconds: float = 120.0

    default_profile: str = "laptop-cpu"
    max_workers: int = 1
    max_agent_steps: int = 6
    task_timeout_seconds: float = 120.0
    review_timeout_seconds: float = 180.0
    allow_direct_ollama_fallback: bool = False

    otel_enabled: bool = False
    otel_endpoint: str | None = "http://otel-collector:4317"
    otel_service_name: str = "private-ai-stack"

    audit_dir: Path = Path("audit")
    reports_dir: Path = Path("reports")
    export_dir: Path = Path("exports")
    max_review_file_bytes: int = 250_000
    max_document_bytes: int = 1_000_000
    max_document_chunks: int = 1_000
    max_audit_record_bytes: int = 64_000
    max_request_id_chars: int = 128
    max_static_tool_output_bytes: int = 20_000
    allow_hosted_providers: bool = False

    embedding_dimensions: int = 64

    @field_validator("api_host", "ollama_model", "default_profile", "otel_service_name")
    @classmethod
    def validate_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("api_port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("must be between 1 and 65535")
        return value

    @field_validator("max_workers", "max_agent_steps", "max_document_chunks")
    @classmethod
    def validate_positive_limits(cls, value: int) -> int:
        if not 1 <= value <= 1_000:
            raise ValueError("must be between 1 and 1000")
        return value

    @field_validator("ollama_timeout_seconds", "task_timeout_seconds", "review_timeout_seconds")
    @classmethod
    def validate_timeouts(cls, value: float) -> float:
        if not 0 < value <= 3_600:
            raise ValueError("must be greater than zero and at most 3600 seconds")
        return value

    @field_validator("max_review_file_bytes", "max_document_bytes", "max_audit_record_bytes", "max_static_tool_output_bytes")
    @classmethod
    def validate_byte_limits(cls, value: int) -> int:
        if not 1 <= value <= 50_000_000:
            raise ValueError("must be between 1 and 50000000 bytes")
        return value

    @field_validator("max_request_id_chars")
    @classmethod
    def validate_request_id_limit(cls, value: int) -> int:
        if not 8 <= value <= 512:
            raise ValueError("must be between 8 and 512")
        return value

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        if value == "memory://local":
            return value
        parsed = urlparse(value)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or not parsed.path.strip("/"):
            raise ValueError("must be memory://local or a PostgreSQL DSN with host and database name")
        return value

    @field_validator("ollama_base_url", "otel_endpoint")
    @classmethod
    def validate_http_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an absolute HTTP(S) URL")
        return value.rstrip("/")

    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        if len(normalized) < 16:
            raise ValueError("must contain at least 16 characters when enabled")
        return normalized

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        if not 8 <= value <= 4_096:
            raise ValueError("embedding_dimensions must be between 8 and 4096")
        return value

    @model_validator(mode="after")
    def validate_environment_requirements(self) -> "Settings":
        if self.environment not in {"development", "test", "production"}:
            raise ValueError("environment must be development, test, or production")
        if self.environment == "production" and self.api_key is None:
            raise ValueError("api_key is required when environment is production")
        return self


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.audit_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.export_dir.mkdir(parents=True, exist_ok=True)
    return settings
