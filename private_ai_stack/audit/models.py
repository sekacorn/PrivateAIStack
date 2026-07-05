from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from private_ai_stack.api.schemas import utc_now


class AuditRecord(BaseModel):
    event_type: str
    actor: str = "system"
    entity_type: str
    entity_id: str
    request_id: str | None = None
    trace_id: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    details: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str | None = None
    record_hash: str | None = None
