import pytest

from private_ai_stack.config.settings import Settings
from private_ai_stack.memory.store import MemoryStore


class BrokenConnection:
    def close(self) -> None:
        pass

    def cursor(self) -> None:
        raise RuntimeError("database connection lost")


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

    assert await store.status() == "volatile"
    await store.ingest("first", "doc.md", {}, False)
    replaced = await store.ingest("second", "doc.md", {}, True)

    assert replaced[1] == 1


@pytest.mark.asyncio
async def test_memory_search_ties_are_ordered_by_chunk_id(tmp_path) -> None:
    settings = Settings(
        database_url="memory://local",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
    )
    store = MemoryStore(settings)
    await store.ingest("alpha", "b.md", {}, False)
    await store.ingest("beta", "a.md", {}, False)

    hits = await store.search("unmatched-token", 2)

    assert [chunk.chunk_id for chunk, _ in hits] == sorted(chunk.chunk_id for chunk, _ in hits)


@pytest.mark.asyncio
async def test_memory_rejects_documents_above_configured_limit(tmp_path) -> None:
    settings = Settings(
        database_url="memory://local",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
        max_document_bytes=4,
    )
    store = MemoryStore(settings)

    with pytest.raises(ValueError, match="document_too_large"):
        await store.ingest("abcde", "doc.md", {}, False)


@pytest.mark.asyncio
async def test_durable_store_discards_broken_connection_for_later_recovery(tmp_path) -> None:
    settings = Settings(
        database_url="postgresql://user:password@db.example/private",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
    )
    store = MemoryStore(settings)
    store._connection = BrokenConnection()
    store._loaded = True

    assert await store.status() == "unavailable"
    assert store._connection is None
    assert store._loaded is False
