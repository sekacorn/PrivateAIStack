import hashlib
from typing import Literal

from pydantic import BaseModel

Severity = Literal["critical", "high", "medium", "low", "informational"]
Confidence = Literal["confirmed", "high", "medium", "speculative"]


class Finding(BaseModel):
    id: str
    source_tool: str
    agent: str
    category: str
    severity: Severity
    confidence: Confidence
    title: str
    explanation: str
    file: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    evidence: str = ""
    remediation: str = ""
    fingerprint: str
    status: str = "open"


def make_fingerprint(source_tool: str, title: str, file: str | None, line: int | None) -> str:
    raw = f"{source_tool}:{title}:{file or ''}:{line or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


class ToolRun(BaseModel):
    tool: str
    status: Literal["passed", "failed", "not_run"]
    reason: str | None = None
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
