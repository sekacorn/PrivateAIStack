"""Every agent-memory.v1 export this package writes must be readable by the suite.

Same arrangement as ``test_suite_record_contract.py``: the suite is installed
only in the ``suite-contract`` job, which sets ``PRIVATEAISTACK_REQUIRE_SUITE_CONTRACT``
so a missing suite fails instead of skipping. Offline, everything here skips.

Three things are checked, because they fail differently:

* the suite's reader accepts the export, re-checking every digest and binding
  each item to its vector;
* the suite's JSON Schemas accept the document, every item, and the linked
  embedding inventory (``additionalProperties: false`` catches field drift the
  reader may tolerate);
* scoring: on its own the export scores 50 (nothing shows it can be restored).
  Imported into AgentForge (a dependency of this package) with an import
  report bound to it, the same export scores 25 at ``verified (memory export)``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from private_ai_stack.config.settings import Settings
from private_ai_stack.memory.portable import export_agent_memory
from private_ai_stack.memory.store import MemoryStore

REQUIRE_ENV = "PRIVATEAISTACK_REQUIRE_SUITE_CONTRACT"


def _import_or_fail(module: str, package: str) -> Any:
    """Skip offline; fail in the contract job, where a skip would be a false pass."""
    try:
        return importlib.import_module(module)
    except ImportError:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(f"{package} is not installed and {REQUIRE_ENV}=1, so this contract check cannot run.")
        pytest.skip(f"{package} is not installed; set {REQUIRE_ENV}=1 to make this a failure.", allow_module_level=True)


agent_memory = _import_or_fail("ai_infrastructure_suite.agent_memory", "ai-infrastructure-suite")
memory_vector = _import_or_fail("ai_infrastructure_suite.memory_vector", "ai-infrastructure-suite")
jsonschema = _import_or_fail("jsonschema", "jsonschema")

NOW = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
CREDENTIAL = "sk-proj-" + "Q" * 32


def _schema(name: str) -> dict[str, Any]:
    from importlib.resources import files

    return json.loads(files("ai_infrastructure_suite").joinpath(f"data/{name}").read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _export(tmp_path: Path, *, redact: bool = False) -> Path:
    settings = Settings(
        database_url="memory://local",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
    )
    store = MemoryStore(settings)
    await store.ingest("Damaged items are replaced within thirty days.", "returns.md", {"team": "support"}, False)
    await store.ingest("Invoices are issued on the first business day. " * 40, "billing.md", {}, False)
    if redact:
        await store.ingest(f"The staging deploy key is {CREDENTIAL}.", "ops.md", {}, False)
    out = tmp_path / "export"
    await export_agent_memory(store, out, export_id="exp-contract", redact_secrets=redact, now=NOW, shard_items=2)
    return out


def _all_items(out: Path) -> list[dict[str, Any]]:
    document = json.loads((out / "agent-memory.json").read_bytes())
    items: list[dict[str, Any]] = []
    for shard in document["stores"][0]["items"]["shards"]:
        items.extend(json.loads(line) for line in (out / shard["path"]).read_text(encoding="utf-8").splitlines())
    return items


def test_the_installed_suite_is_reported() -> None:
    import ai_infrastructure_suite

    print(f"validated against ai-infrastructure-suite {ai_infrastructure_suite.__version__}")


@pytest.mark.asyncio
@pytest.mark.parametrize("redact", [False, True])
async def test_the_suite_reader_accepts_the_export(tmp_path: Path, redact: bool) -> None:
    out = await _export(tmp_path, redact=redact)
    document = out / "agent-memory.json"

    export = agent_memory.load_agent_memory(document, expected_sha256=_sha(document))

    assert export.source_system == "privateaistack"
    assert export.redaction_profile == ("secrets" if redact else None)
    (store,) = export.stores
    assert store.store_id == "knowledge" and store.deleted_by_policy == 0
    assert store.item_count == len(_all_items(out))


@pytest.mark.asyncio
@pytest.mark.parametrize("redact", [False, True])
async def test_the_suite_schemas_accept_the_export(tmp_path: Path, redact: bool) -> None:
    out = await _export(tmp_path, redact=redact)
    document = json.loads((out / "agent-memory.json").read_bytes())
    schema = _schema("agent-memory.schema.json")
    jsonschema.validate(document, schema)
    # Items live in shards, which the document schema does not reach; the same
    # items inlined as a single_file store must validate too.
    inlined = json.loads(json.dumps(document))
    inlined["stores"][0]["items"] = {"layout": "single_file", "records": _all_items(out)}
    jsonschema.validate(inlined, schema)
    inventory = json.loads((out / "embeddings" / "embedding-portability.json").read_bytes())
    jsonschema.validate(inventory, _schema("embedding-portability.schema.json"))


@pytest.mark.asyncio
async def test_an_unknown_item_field_is_rejected(tmp_path: Path) -> None:
    """The check has teeth: a drifted item fails both halves."""
    out = await _export(tmp_path)
    document = json.loads((out / "agent-memory.json").read_bytes())
    items = _all_items(out)
    items[0]["speculative_new_field"] = "whatever"
    inlined = json.loads(json.dumps(document))
    inlined["stores"][0]["items"] = {"layout": "single_file", "records": items}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(inlined, _schema("agent-memory.schema.json"))
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(inlined), encoding="utf-8")
    with pytest.raises(ValueError):
        agent_memory.load_agent_memory(path)


def _manifest(root: Path, export_dir: Path, report: Path | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "state": "memory_inventory",
        "path": (export_dir / "agent-memory.json").relative_to(root).as_posix(),
        "sha256": _sha(export_dir / "agent-memory.json"),
    }
    if report is not None:
        entry["import_report"] = {"path": report.relative_to(root).as_posix(), "sha256": _sha(report)}
    return entry


@pytest.mark.asyncio
async def test_the_export_alone_scores_50(tmp_path: Path) -> None:
    out = await _export(tmp_path)

    result = memory_vector.build_vector(tmp_path, _manifest(tmp_path, out, None))

    (row,) = result.section
    assert row["score"] == 50 and row["verification_level"] == "verified (memory export)"
    assert row["embeddings"]["items_without_vector"] == 0
    assert "no import report" in row["reason"]


@pytest.mark.asyncio
async def test_imported_into_agentforge_the_export_scores_25(tmp_path: Path) -> None:
    from forge.memory.transfer import import_memory
    from forge.memory.vector import InMemoryVectorStore

    out = await _export(tmp_path, redact=True)
    report = tmp_path / "import-report.json"
    summary = await import_memory(
        InMemoryVectorStore(),
        out / "agent-memory.json",
        report_path=report,
        destination_backend="in_memory",
        destination_store_id="kb",
    )
    assert summary.refused == {} and summary.imported == summary.received

    result = memory_vector.build_vector(tmp_path, _manifest(tmp_path, out, report))

    (row,) = result.section
    assert row["score"] == 25 and row["verification_level"] == "verified (memory export)"
    assert row["embeddings"]["items_bound"] == summary.received
    assert row["import"]["refused"] == 0
    assert result.row["score"] == 25
