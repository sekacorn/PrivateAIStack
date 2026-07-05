import hashlib
import json
from pathlib import Path
from typing import Any

from private_ai_stack.audit.models import AuditRecord
from private_ai_stack.audit.redaction import redact


class AuditWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str | None:
        if not self.path.exists():
            return None
        last = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = json.loads(line).get("record_hash")
        return str(last) if last else None

    def write(
        self,
        event_type: str,
        *,
        entity_type: str,
        entity_id: str,
        actor: str = "system",
        request_id: str | None = None,
        trace_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuditRecord:
        record = AuditRecord(
            event_type=event_type,
            actor=actor,
            entity_type=entity_type,
            entity_id=entity_id,
            request_id=request_id,
            trace_id=trace_id,
            details=redact(details or {}),
            previous_hash=self._last_hash,
        )
        payload = record.model_dump(mode="json", exclude={"record_hash"})
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        record.record_hash = digest
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        self._last_hash = digest
        return record

    def records_for(self, entity_id: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("entity_id") == entity_id:
                records.append(item)
        return records

    def export_jsonl(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            target.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            target.write_text("", encoding="utf-8")
        return target
