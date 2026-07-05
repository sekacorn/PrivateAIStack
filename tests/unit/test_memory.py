import pytest

from private_ai_stack.config.settings import Settings
from private_ai_stack.memory.store import MemoryStore


@pytest.mark.asyncio
async def test_memory_idempotent_ingestion_and_search(tmp_path) -> None:
    settings = Settings(
        database_url="memory://local", audit_dir=tmp_path / "audit", reports_dir=tmp_path / "reports", export_dir=tmp_path / "exports"
    )
    store = MemoryStore(settings)

    first = await store.ingest("alpha beta audit records", "doc.md", {}, False)
    second = await store.ingest("alpha beta audit records", "doc.md", {}, False)
    hits = await store.search("audit", 1)

    assert first[1] == 1
    assert second[2] is True
    assert hits[0][0].source_name == "doc.md"

    export = await store.export_jsonl(str(tmp_path / "memory.jsonl"))
    assert "memory.jsonl" in export


@pytest.mark.asyncio
async def test_memory_replace_existing_and_status(tmp_path) -> None:
    settings = Settings(
        database_url="memory://local",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
    )
    store = MemoryStore(settings)

    assert await store.status() == "ok"
    await store.ingest("first", "doc.md", {}, False)
    replaced = await store.ingest("second", "doc.md", {}, True)

    assert replaced[1] == 1
