from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class Status(StrEnum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class HealthResponse(BaseModel):
    status: str
    service: str = "PrivateAIStack"
    timestamp: datetime = Field(default_factory=utc_now)


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, str]
    timestamp: datetime = Field(default_factory=utc_now)


class VersionResponse(BaseModel):
    name: str = "PrivateAIStack"
    version: str
    forge_version: str
    default_model: str


class TaskRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=8000)
    actor: str = "local-user"


class TaskResponse(BaseModel):
    task_id: str
    status: Status
    request_id: str
    trace_id: str | None = None
    created_at: datetime
    updated_at: datetime
    result: dict[str, Any] | None = None
    error: str | None = None


class KnowledgeDocumentRequest(BaseModel):
    content: str = Field(min_length=1)
    source_name: str = "inline-document"
    metadata: dict[str, Any] = Field(default_factory=dict)
    replace_existing: bool = False


class KnowledgeDocumentResponse(BaseModel):
    document_id: str
    chunks_created: int
    idempotent: bool
    content_hash: str


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class KnowledgeHit(BaseModel):
    document_id: str
    chunk_id: str
    source_name: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchResponse(BaseModel):
    query: str
    hits: list[KnowledgeHit]


ReviewMode = Literal["safe-static", "sandboxed-execution"]


class ReviewRequest(BaseModel):
    repository_path: str
    mode: ReviewMode = "safe-static"
    languages: list[str] = Field(default_factory=lambda: ["auto"])
    include_agents: list[str] = Field(
        default_factory=lambda: [
            "code_quality",
            "security",
            "tests",
            "infrastructure",
            "documentation",
        ]
    )


class ReviewResponse(BaseModel):
    review_id: str
    status: Status
    request_id: str
    trace_id: str | None = None
    created_at: datetime
    updated_at: datetime
    summary: dict[str, Any] | None = None
    error: str | None = None
