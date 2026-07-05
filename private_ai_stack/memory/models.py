from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from private_ai_stack.api.schemas import utc_now


class DocumentChunk(BaseModel):
    document_id: str
    chunk_id: str
    source_name: str
    text: str
    content_hash: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
