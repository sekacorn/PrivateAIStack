from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from private_ai_stack.audit.writer import AuditWriter


def test_audit_redacts_and_hash_chains(tmp_path: Path) -> None:
    writer = AuditWriter(tmp_path / "audit.jsonl")
    first = writer.write("event.one", entity_type="task", entity_id="1", details={"password": "password=super-secret"})
    second = writer.write("event.two", entity_type="task", entity_id="1", details={"email": "person@example.com"})

    records = writer.records_for("1")
    assert len(records) == 2
    assert "[REDACTED]" in records[0]["details"]["password"]
    assert "[REDACTED]" in records[1]["details"]["email"]
    assert second.previous_hash == first.record_hash


def test_audit_redacts_sensitive_values_by_key(tmp_path: Path) -> None:
    writer = AuditWriter(tmp_path / "audit.jsonl")
    writer.write(
        "event.one",
        entity_type="task",
        entity_id="1",
        details={"password": "super-secret", "nested": {"api_key": "abc123"}, "safe": "ok"},
    )

    record = writer.records_for("1")[0]
    assert record["details"]["password"] == "[REDACTED]"  # noqa: S105 - expected redaction marker
    assert record["details"]["nested"]["api_key"] == "[REDACTED]"
    assert record["details"]["safe"] == "ok"


def test_audit_loads_previous_hash_and_exports(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    writer = AuditWriter(path)
    first = writer.write("event.one", entity_type="task", entity_id="1")
    second_writer = AuditWriter(path)
    second = second_writer.write("event.two", entity_type="task", entity_id="2")
    export = second_writer.export_jsonl(tmp_path / "exports" / "audit.jsonl")

    assert second.previous_hash == first.record_hash
    assert export.exists()
    assert second_writer.records_for("missing") == []


def test_audit_concurrent_writers_keep_one_valid_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"

    def append(index: int) -> None:
        AuditWriter(path).write("event.concurrent", entity_type="task", entity_id=str(index))

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(24)))

    valid, count, reason = AuditWriter(path).verify()
    assert valid, reason
    assert count == 24


def test_audit_rejects_malformed_existing_log(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    valid, count, reason = AuditWriter(path, verify_existing=False).verify()

    assert not valid
    assert count == 0
    assert reason and "not valid JSON" in reason
