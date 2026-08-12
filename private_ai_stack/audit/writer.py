import hashlib
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from private_ai_stack.audit.models import AuditRecord
from private_ai_stack.audit.redaction import redact

_path_locks: dict[Path, threading.RLock] = {}
_path_locks_guard = threading.Lock()


class AuditIntegrityError(RuntimeError):
    """Raised when an existing audit file cannot safely extend its hash chain."""


class AuditWriter:
    """Append redacted local audit records with a process-coordinated hash chain.

    The lock coordinates cooperating writers on one host filesystem. The chain is
    tamper-evident, not immutable evidence or a substitute for external anchoring.
    """

    def __init__(self, path: Path, max_record_bytes: int = 64_000, verify_existing: bool = True) -> None:
        self.path = path.resolve()
        self.max_record_bytes = max_record_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._load_last_hash() if verify_existing else None

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with _path_locks_guard:
            lock = _path_locks.setdefault(self.path, threading.RLock())
        with lock:
            lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
            with lock_path.open("a+b") as handle:
                self._lock_file(handle)
                try:
                    yield
                finally:
                    self._unlock_file(handle)

    @staticmethod
    def _lock_file(handle: Any) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        except OSError as exc:
            raise AuditIntegrityError("Unable to acquire the local audit lock.") from exc

    @staticmethod
    def _unlock_file(handle: Any) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        except OSError:
            pass

    def _records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(line.encode("utf-8")) > self.max_record_bytes:
                    raise AuditIntegrityError(f"Audit record on line {line_number} exceeds the configured size limit.")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditIntegrityError(f"Audit record on line {line_number} is not valid JSON.") from exc
                if not isinstance(value, dict):
                    raise AuditIntegrityError(f"Audit record on line {line_number} is not an object.")
                records.append(value)
        return records

    @staticmethod
    def _record_hash(record: dict[str, Any]) -> str:
        payload = dict(record)
        payload.pop("record_hash", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_last_hash(self) -> str | None:
        records = self._records()
        if not records:
            return None
        valid, _, reason = self.verify()
        if not valid:
            raise AuditIntegrityError(reason or "Audit hash chain is invalid.")
        record_hash = records[-1].get("record_hash")
        return str(record_hash) if record_hash else None

    def verify(self) -> tuple[bool, int, str | None]:
        try:
            records = self._records()
        except AuditIntegrityError as exc:
            return False, 0, str(exc)
        previous: str | None = None
        for index, record in enumerate(records, start=1):
            if record.get("previous_hash") != previous:
                return False, index, f"Audit hash chain mismatch on line {index}."
            actual = self._record_hash(record)
            if actual != record.get("record_hash"):
                return False, index, f"Audit record hash mismatch on line {index}."
            previous = actual
        return True, len(records), None

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
        with self._locked():
            self._last_hash = self._load_last_hash()
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
            record.record_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            serialized = record.model_dump_json() + "\n"
            if len(serialized.encode("utf-8")) > self.max_record_bytes:
                raise AuditIntegrityError("Audit record exceeds the configured size limit.")
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            self._last_hash = record.record_hash
            return record

    def records_for(self, entity_id: str) -> list[dict[str, Any]]:
        with self._locked():
            return [record for record in self._records() if record.get("entity_id") == entity_id]

    def export_jsonl(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            target.write_text(self.path.read_text(encoding="utf-8") if self.path.exists() else "", encoding="utf-8")
        return target
