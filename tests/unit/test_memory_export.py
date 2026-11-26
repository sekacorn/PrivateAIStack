"""The agent-memory.v1 writer: shape, bindings, and every refusal."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.cli.main import app
from private_ai_stack.config.settings import Settings
from private_ai_stack.memory.embeddings import DeterministicEmbeddingModel
from private_ai_stack.memory.portable import (
    DOCUMENT_NAME,
    MemoryExportError,
    export_agent_memory,
)
from private_ai_stack.memory.store import MemoryStore
from private_ai_stack.reembed.embedders import PRIVATEAISTACK_HASHED_BOW

NOW = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
# Built at run time so no scanner mistakes the fixture for a real key.
CREDENTIAL = "sk-proj-" + "Q" * 32


def _settings(tmp_path: Path, dimensions: int = 64) -> Settings:
    return Settings(
        database_url="memory://local",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
        embedding_dimensions=dimensions,
    )


async def _store(tmp_path: Path, *documents: tuple[str, str, dict[str, Any]], dimensions: int = 64) -> MemoryStore:
    store = MemoryStore(_settings(tmp_path, dimensions))
    for content, source, metadata in documents:
        await store.ingest(content, source, metadata, False)
    return store


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _items(out: Path, document: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for shard in document["stores"][0]["items"]["shards"]:
        data = (out / shard["path"]).read_bytes()
        assert _sha(data) == shard["sha256"]
        lines = data.decode("utf-8").splitlines()
        assert len(lines) == shard["record_count"]
        items.extend(json.loads(line) for line in lines)
    return items


@pytest.mark.asyncio
async def test_export_writes_bound_items_and_vectors(tmp_path: Path) -> None:
    store = await _store(
        tmp_path,
        ("Damaged items are replaced within thirty days.", "returns.md", {"team": "support"}),
        ("x" * 1500, "long.md", {}),
    )
    out = tmp_path / "export"
    summary = await export_agent_memory(store, out, export_id="exp-1", now=NOW)

    document = json.loads((out / DOCUMENT_NAME).read_bytes())
    assert summary.sha256 == _sha((out / DOCUMENT_NAME).read_bytes())
    assert summary.items == 3 and summary.embeddings and summary.excluded == ()
    assert document["schema_version"] == "agent-memory.v1"
    assert document["generated_by"]["tool"] == "privateaistack"
    assert document["export"] == {
        "export_id": "exp-1",
        "created_at": "2026-11-25T12:00:00Z",
        "redaction_profile": None,
        "source": {"system": "privateaistack", "backend": "volatile"},
    }
    store_doc = document["stores"][0]
    assert store_doc["store_id"] == "knowledge" and store_doc["kind"] == "retrieval"
    assert store_doc["counts"] == {"active": 3, "quarantined": 0, "deleted_by_policy": 0}
    assert store_doc["tombstones"] == []

    chunks = {chunk.chunk_id: chunk for chunk in await store.all_chunks()}
    items = _items(out, document)
    assert [item["item_id"] for item in items] == sorted(chunks)
    for item in items:
        chunk = chunks[item["item_id"]]
        assert item["text"] == chunk.text
        assert item["content_sha256"] == _sha(chunk.text.encode("utf-8"))
        assert item["status"] == "active" and item["provenance"] is None
        assert item["metadata"] == {
            "document_id": chunk.document_id,
            "source_name": chunk.source_name,
            "document_sha256": chunk.content_hash,
            "metadata": chunk.metadata,
        }
        assert item["created_at"].endswith("Z")

    link = store_doc["embeddings"]
    assert link["inventory_store_id"] == "knowledge"
    inventory_bytes = (out / link["inventory"]["path"]).read_bytes()
    assert _sha(inventory_bytes) == link["inventory"]["sha256"]
    inventory = json.loads(inventory_bytes)
    assert inventory["schema_version"] == "embedding-portability.v1"
    model = inventory["stores"][0]["embedding_model"]
    assert (model["id"], model["revision"], model["dimensions"]) == (PRIVATEAISTACK_HASHED_BOW, "1", 64)
    vectors = inventory["stores"][0]["vectors"]
    root = out / "embeddings"
    by_id = {item["item_id"]: item for item in items}
    seen = []
    for shard in vectors["shards"]:
        data = (root / shard["path"]).read_bytes()
        assert _sha(data) == shard["sha256"]
        for line in data.decode("utf-8").splitlines():
            record = json.loads(line)
            item = by_id[record["record_id"]]
            assert record["source_text_sha256"] == item["content_sha256"]
            assert record["values"] == chunks[record["record_id"]].embedding
            seen.append(record["record_id"])
    assert seen == [item["item_id"] for item in items]
    for entry in vectors["source_text"]:
        source = root / entry["source_text"]["path"]
        assert _sha(source.read_bytes()) == entry["source_text"]["sha256"] == by_id[entry["record_id"]]["content_sha256"]


@pytest.mark.asyncio
async def test_export_is_deterministic_and_shards(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha beta", "a.md", {}), ("gamma delta", "b.md", {}), ("epsilon", "c.md", {}))
    first = await export_agent_memory(store, tmp_path / "one", export_id="exp-1", now=NOW, shard_items=2)
    second = await export_agent_memory(store, tmp_path / "two", export_id="exp-1", now=NOW, shard_items=2)
    assert first.sha256 == second.sha256
    document = json.loads(first.path.read_bytes())
    assert [shard["record_count"] for shard in document["stores"][0]["items"]["shards"]] == [2, 1]


@pytest.mark.asyncio
async def test_empty_store_exports_single_file_without_embeddings(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    summary = await export_agent_memory(store, tmp_path / "out", now=NOW)
    document = json.loads(summary.path.read_bytes())
    assert document["stores"][0]["items"] == {"layout": "single_file", "records": []}
    assert document["stores"][0]["embeddings"] is None
    assert summary.export_id == "exp-20261125T120000Z" and not summary.embeddings


@pytest.mark.asyncio
async def test_no_embeddings_option(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", {}))
    summary = await export_agent_memory(store, tmp_path / "out", include_embeddings=False, now=NOW)
    assert json.loads(summary.path.read_bytes())["stores"][0]["embeddings"] is None
    assert not (tmp_path / "out" / "embeddings").exists()


@pytest.mark.asyncio
async def test_credential_stops_the_export_and_is_never_shown(tmp_path: Path) -> None:
    store = await _store(tmp_path, (f"deploy key {CREDENTIAL}", "ops.md", {}))
    chunk_id = (await store.all_chunks())[0].chunk_id
    out = tmp_path / "out"
    with pytest.raises(MemoryExportError) as caught:
        await export_agent_memory(store, out, now=NOW)
    message = str(caught.value)
    assert chunk_id in message and "api_key_sk_prefix" in message
    assert CREDENTIAL not in message
    assert "--redact-secrets" in message and "--exclude" in message
    assert not out.exists()


@pytest.mark.asyncio
async def test_credential_in_metadata_is_refused(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("harmless text", "ops.md", {"note": CREDENTIAL}))
    with pytest.raises(MemoryExportError, match="credential"):
        await export_agent_memory(store, tmp_path / "out", now=NOW)


@pytest.mark.asyncio
async def test_redact_secrets_binds_vectors_to_the_redacted_text(tmp_path: Path) -> None:
    store = await _store(tmp_path, (f"deploy key {CREDENTIAL} for staging", "ops.md", {"note": CREDENTIAL}))
    out = tmp_path / "out"
    summary = await export_agent_memory(store, out, redact_secrets=True, now=NOW)
    document = json.loads(summary.path.read_bytes())
    assert document["export"]["redaction_profile"] == "secrets"
    raw = b"".join(path.read_bytes() for path in out.rglob("*") if path.is_file())
    assert CREDENTIAL.encode() not in raw
    (item,) = _items(out, document)
    assert "[REDACTED:api_key_sk_prefix]" in item["text"]
    assert item["metadata"]["metadata"]["note"] == "[REDACTED:api_key_sk_prefix]"
    inventory = json.loads((out / "embeddings" / "embedding-portability.json").read_bytes())
    shard = inventory["stores"][0]["vectors"]["shards"][0]
    record = json.loads((out / "embeddings" / shard["path"]).read_text(encoding="utf-8"))
    assert record["source_text_sha256"] == item["content_sha256"]
    assert record["values"] == DeterministicEmbeddingModel(64).embed(item["text"])


@pytest.mark.asyncio
async def test_exclusion_is_audited_and_leaves_the_chunk_out(tmp_path: Path) -> None:
    store = await _store(tmp_path, (f"deploy key {CREDENTIAL}", "ops.md", {}), ("keep me", "keep.md", {}))
    flagged = next(c.chunk_id for c in await store.all_chunks() if c.source_name == "ops.md")
    audit = AuditWriter(tmp_path / "audit" / "audit.jsonl")
    summary = await export_agent_memory(
        store, tmp_path / "out", exclude={flagged: "false positive: sample key in a tutorial"}, audit=audit, now=NOW
    )
    assert summary.excluded == (flagged,) and summary.items == 1
    document = json.loads(summary.path.read_bytes())
    assert flagged not in {item["item_id"] for item in _items(tmp_path / "out", document)}
    (record,) = audit.records_for(flagged)
    assert record["event_type"] == "memory.export_exclusion"
    assert record["details"]["export_sha256"] == summary.sha256
    assert record["details"]["reason"] == "false positive: sample key in a tutorial"
    assert (await store.all_chunks())[0].chunk_id  # excluding never deletes


@pytest.mark.asyncio
async def test_exclusion_needs_a_reason_an_audit_log_and_a_known_chunk(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", {}))
    chunk_id = (await store.all_chunks())[0].chunk_id
    audit = AuditWriter(tmp_path / "audit" / "audit.jsonl")
    with pytest.raises(MemoryExportError, match="needs a reason"):
        await export_agent_memory(store, tmp_path / "a", exclude={chunk_id: " "}, audit=audit)
    with pytest.raises(MemoryExportError, match="audit log"):
        await export_agent_memory(store, tmp_path / "b", exclude={chunk_id: "reason"})
    with pytest.raises(MemoryExportError, match="does not hold: nope"):
        await export_agent_memory(store, tmp_path / "c", exclude={"nope": "reason"}, audit=audit)
    assert not any((tmp_path / name).exists() for name in "abc")


@pytest.mark.asyncio
async def test_an_inconsistent_index_is_refused(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha beta", "a.md", {}), ("gamma", "b.md", {}))
    store._chunks[0] = store._chunks[0].model_copy(update={"embedding": [0.0] * 32})
    with pytest.raises(MemoryExportError) as caught:
        await export_agent_memory(store, tmp_path / "out", now=NOW)
    assert store._chunks[0].chunk_id in str(caught.value) and "dimension 64" in str(caught.value)
    assert not (tmp_path / "out").exists()
    # Without vectors there is no binding to claim, so the text still exports.
    summary = await export_agent_memory(store, tmp_path / "text-only", include_embeddings=False, now=NOW)
    assert summary.items == 2


@pytest.mark.asyncio
async def test_a_changed_vector_value_is_refused(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha beta", "a.md", {}))
    values = list(store._chunks[0].embedding)
    values[0] += 0.5
    store._chunks[0] = store._chunks[0].model_copy(update={"embedding": values})
    with pytest.raises(MemoryExportError, match="not this store's embedding"):
        await export_agent_memory(store, tmp_path / "out", now=NOW)


@pytest.mark.asyncio
async def test_target_must_be_an_empty_directory(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", {}))
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(MemoryExportError, match="empty directory"):
        await export_agent_memory(store, busy, now=NOW)
    a_file = tmp_path / "file"
    a_file.write_text("x", encoding="utf-8")
    with pytest.raises(MemoryExportError, match="empty directory"):
        await export_agent_memory(store, a_file, now=NOW)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (await export_agent_memory(store, empty, now=NOW)).items == 1


@pytest.mark.asyncio
async def test_a_linked_target_is_refused(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", {}))
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(real, link, target_is_directory=True)
    except OSError:
        pytest.skip("This platform does not allow creating a symlink here.")
    with pytest.raises(MemoryExportError, match="link"):
        await export_agent_memory(store, link, now=NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"store_id": "has space"}, "store ID"),
        ({"export_id": "-leading"}, "export ID"),
        ({"shard_items": 0}, "shard_items"),
    ],
)
async def test_invalid_arguments(tmp_path: Path, kwargs: dict[str, Any], match: str) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", {}))
    with pytest.raises(MemoryExportError, match=match):
        await export_agent_memory(store, tmp_path / "out", **kwargs)


def _nested(depth: int) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for _ in range(depth):
        value = {"k": value}
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        (_nested(8), "deeper"),
        ({f"k{i}": i for i in range(65)}, "more than 64 keys"),
        ({"k": list(range(257))}, "more than 256 items"),
        ({"k": "x" * 4097}, "over 4096"),
        ({"k": float("nan")}, "non-finite"),
    ],
)
async def test_metadata_outside_the_reader_bounds_is_refused(tmp_path: Path, metadata: dict[str, Any], match: str) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", metadata))
    with pytest.raises(MemoryExportError, match=match):
        await export_agent_memory(store, tmp_path / "out", now=NOW)
    assert not (tmp_path / "out").exists()


@pytest.mark.asyncio
async def test_metadata_that_is_not_json_is_refused(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", {}))
    store._chunks[0] = store._chunks[0].model_copy(update={"metadata": {"k": object()}})
    with pytest.raises(MemoryExportError, match="not JSON"):
        await export_agent_memory(store, tmp_path / "out", now=NOW)


@pytest.mark.asyncio
async def test_a_naive_timestamp_is_refused(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", {}))
    store._chunks[0] = store._chunks[0].model_copy(update={"created_at": datetime(2026, 1, 1)})  # noqa: DTZ001
    with pytest.raises(MemoryExportError, match="time zone"):
        await export_agent_memory(store, tmp_path / "out", now=NOW)


@pytest.mark.asyncio
async def test_export_jsonl_still_works_and_says_it_is_superseded(tmp_path: Path) -> None:
    store = await _store(tmp_path, ("alpha", "a.md", {}))
    with pytest.warns(DeprecationWarning, match="export_agent_memory"):
        path = await store.export_jsonl(str(tmp_path / "memory.jsonl"))
    assert json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])["source_name"] == "a.md"


def _cli_env(tmp_path: Path) -> dict[str, str]:
    return {
        "DATABASE_URL": "memory://local",
        "AUDIT_DIR": str(tmp_path / "audit"),
        "REPORTS_DIR": str(tmp_path / "reports"),
        "EXPORT_DIR": str(tmp_path / "exports"),
    }


def test_cli_exports_the_configured_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["memory", "export", str(tmp_path / "out"), "--export-id", "exp-cli"], env=_cli_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "Exported 0 chunk(s)" in result.output
    document = json.loads((tmp_path / "out" / DOCUMENT_NAME).read_bytes())
    assert document["export"]["export_id"] == "exp-cli"


def test_cli_exclude_needs_a_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["memory", "export", str(tmp_path / "out"), "--exclude", "x"], env=_cli_env(tmp_path))
    assert result.exit_code == 2
    assert "--exclude needs --exclude-reason" in result.output


def test_cli_refusal_is_invalid_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app,
        ["memory", "export", str(tmp_path / "out"), "--exclude", "missing", "--exclude-reason", "r"],
        env=_cli_env(tmp_path),
    )
    assert result.exit_code == 2
    assert "does not hold: missing" in result.output
    assert not (tmp_path / "out").exists()
