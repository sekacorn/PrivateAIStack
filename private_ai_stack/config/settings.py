from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "PrivateAIStack"
    environment: str = "development"
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

    otel_enabled: bool = False
    otel_endpoint: str | None = "http://otel-collector:4317"
    otel_service_name: str = "private-ai-stack"

    audit_dir: Path = Path("audit")
    reports_dir: Path = Path("reports")
    export_dir: Path = Path("exports")
    max_review_file_bytes: int = 250_000
    allow_hosted_providers: bool = False

    embedding_dimensions: int = 64

    @field_validator("max_workers")
    @classmethod
    def validate_workers(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_workers must be at least 1")
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def validate_embedding_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("embedding_dimensions must be positive")
        return value


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.audit_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    settings.export_dir.mkdir(parents=True, exist_ok=True)
    return settings
