"""Export the knowledge store as ``agent-memory.v1``, verifiably.

The AI Infrastructure Suite owns the format's JSON Schema and scores exports
for the ``memory`` lock-in vector. This module only writes the documents and
never imports the suite; the ``suite-contract`` job feeds what it writes to the
suite's own readers.

An export goes into an empty directory::

    agent-memory.json                       the document (digests of everything below)
    items/0000.jsonl ...                    one chunk per line, canonical bytes
    embeddings/embedding-portability.json   the vectors
    embeddings/vectors/0000.jsonl ...
    embeddings/sources/<sha256>.txt         each chunk's exact text, once per text

Every chunk becomes one retrieval item whose ``item_id`` is its ``chunk_id``.
The document it came from is kept in the item's metadata (``document_id``,
``source_name``, ``document_sha256``); the operator's own document metadata
sits under ``metadata`` so it can never collide with those keys.

**Vectors are bound to the exported text.** Before anything is written, every
stored vector is compared with the store's own deterministic embedding of its
chunk text. A vector that does not match (a changed dimension, a mixed index)
stops the export: exporting it would claim a binding that is not there. The
vectors written are then computed over exactly the exported text, so they stay
bound even when credential redaction changed a chunk.

**A credential is never exported.** A chunk that looks like it holds one stops
the export, and the error names the chunk and the kind of credential, never the
value. To proceed, redact detected credentials (``redact_secrets=True``) or
leave named chunks out (``exclude``, each with a reason, recorded in the audit
log). The detector is AgentForge's, so both writers refuse the same things.

**No tombstones.** This store replaces documents by deleting their chunks and
keeps no deletion marker, so an export always reports ``deleted_by_policy: 0``
and cannot show that anything was deleted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from forge.security.secrets import redact_secrets_in, secret_kinds_in

from private_ai_stack import __version__
from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.memory.models import DocumentChunk
from private_ai_stack.memory.store import MemoryStore
from private_ai_stack.reembed.embedders import PRIVATEAISTACK_HASHED_BOW

MEMORY_SCHEMA_VERSION: Final = "agent-memory.v1"
EMBEDDING_SCHEMA_VERSION: Final = "embedding-portability.v1"
DOCUMENT_NAME: Final = "agent-memory.json"
SOURCE_SYSTEM: Final = "privateaistack"
DEFAULT_STORE_ID: Final = "knowledge"
DEFAULT_SHARD_ITEMS: Final = 1000
EMBEDDING_REVISION: Final = "1"
"""Matches the built-in re-embedding registry's revision for this scheme."""

MAX_ITEMS: Final = 100_000
MAX_TEXT_BYTES: Final = 48 * 1024
MAX_METADATA_DEPTH: Final = 8
MAX_METADATA_KEYS: Final = 64
MAX_METADATA_ITEMS: Final = 256
MAX_METADATA_STRING: Final = 4096
VECTOR_TOLERANCE: Final = 1e-9
EXCLUSION_EVENT: Final = "memory.export_exclusion"

_IDENTIFIER: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


class MemoryExportError(ValueError):
    """An export was refused. Nothing was written, and the message never repeats chunk text."""


@dataclass(frozen=True)
class ExportSummary:
    """What an export wrote."""

    path: Path
    sha256: str
    export_id: str
    items: int
    embeddings: bool
    excluded: tuple[str, ...] = ()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise MemoryExportError("A chunk has a timestamp without a time zone.")
    value = value.astimezone(UTC)
    text = value.strftime("%Y-%m-%dT%H:%M:%S")
    if value.microsecond:
        text += f".{value.microsecond:06d}"
    return text + "Z"


def _identifier(value: str, what: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise MemoryExportError(f"A {what} is not a portable identifier ([A-Za-z0-9][A-Za-z0-9_.:-]{{0,127}}).")
    return value


def _line(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _create(path: Path, data: bytes) -> None:
    """Write a new file; never follows or overwrites anything already there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _write_json(path: Path, document: Mapping[str, Any]) -> str:
    data = (json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _create(path, data)
    return _sha(data)


def _shards(lines: Sequence[bytes], root: Path, prefix: str, size: int) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for index in range(0, len(lines), size):
        chunk = lines[index : index + size]
        data = b"".join(chunk)
        name = f"{prefix}/{index // size:04d}.jsonl"
        _create(root / name, data)
        shards.append({"path": name, "sha256": _sha(data), "record_count": len(chunk)})
    return shards


def _check_metadata(chunk_id: str, value: Any, depth: int = 1) -> None:
    """The reader's metadata bounds, checked here so a bad chunk fails before anything is written."""
    if depth > MAX_METADATA_DEPTH:
        raise MemoryExportError(f"Chunk {chunk_id} has metadata nested deeper than {MAX_METADATA_DEPTH} levels.")
    if isinstance(value, dict):
        if len(value) > MAX_METADATA_KEYS:
            raise MemoryExportError(f"Chunk {chunk_id} has a metadata object with more than {MAX_METADATA_KEYS} keys.")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > MAX_METADATA_STRING:
                raise MemoryExportError(f"Chunk {chunk_id} has an invalid metadata key.")
            _check_metadata(chunk_id, item, depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_METADATA_ITEMS:
            raise MemoryExportError(f"Chunk {chunk_id} has a metadata list with more than {MAX_METADATA_ITEMS} items.")
        for item in value:
            _check_metadata(chunk_id, item, depth + 1)
    elif isinstance(value, str):
        if len(value) > MAX_METADATA_STRING:
            raise MemoryExportError(f"Chunk {chunk_id} has a metadata string over {MAX_METADATA_STRING} characters.")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise MemoryExportError(f"Chunk {chunk_id} has a non-finite number in its metadata.")
    elif value is not None and not isinstance(value, bool | int):
        raise MemoryExportError(f"Chunk {chunk_id} has a metadata value that is not JSON.")


def _refuse_credentials(chunk_id: str, *values: Any) -> None:
    kinds = secret_kinds_in(*values)
    if kinds:
        raise MemoryExportError(
            f"Chunk {chunk_id} appears to hold a credential ({', '.join(kinds)}). Nothing was written. "
            "To proceed: redact detected credentials (--redact-secrets), leave this chunk out of the "
            f"export (--exclude {chunk_id} --exclude-reason '...'), or remove the document from the "
            "store. A credential is never exported."
        )


def _mismatched_vectors(store: MemoryStore, chunks: Sequence[DocumentChunk]) -> list[str]:
    """Chunk IDs whose stored vector is not the store's own embedding of their text."""
    bad: list[str] = []
    for chunk in chunks:
        expected = store.embedding_model.embed(chunk.text)
        if len(chunk.embedding) != len(expected) or any(
            not math.isfinite(stored) or abs(stored - value) > VECTOR_TOLERANCE
            for stored, value in zip(chunk.embedding, expected, strict=False)
        ):
            bad.append(chunk.chunk_id)
    return bad


def _item(chunk: DocumentChunk, redact_secrets: bool) -> dict[str, Any]:
    chunk_id = _identifier(chunk.chunk_id, "chunk ID")
    text = chunk.text
    metadata: dict[str, Any] = {
        "document_id": chunk.document_id,
        "source_name": chunk.source_name,
        "document_sha256": chunk.content_hash,
        "metadata": chunk.metadata,
    }
    if redact_secrets:
        text = redact_secrets_in(text)
        metadata = redact_secrets_in(metadata)
    _refuse_credentials(chunk_id, text, metadata)
    if not text:
        raise MemoryExportError(f"Chunk {chunk_id} has no text.")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise MemoryExportError(f"Chunk {chunk_id} is over 48 KiB.")
    _check_metadata(chunk_id, metadata)
    return {
        "item_id": chunk_id,
        "kind": "document",
        "text": text,
        "content_sha256": _sha(text.encode("utf-8")),
        "metadata": metadata,
        "created_at": _timestamp(chunk.created_at),
        "status": "active",
        "provenance": None,
    }


def _embedding_inventory(store: MemoryStore, items: Sequence[dict[str, Any]], root: Path, store_id: str, size: int) -> str:
    """Write the linked embedding-portability.v1 document and return its digest."""
    directory = root / "embeddings"
    vector_lines: list[bytes] = []
    sources: list[dict[str, Any]] = []
    for item in items:
        digest = item["content_sha256"]
        source = directory / "sources" / f"{digest}.txt"
        if not source.exists():
            _create(source, item["text"].encode("utf-8"))
        vector_lines.append(
            _line(
                {
                    "record_id": item["item_id"],
                    "source_text_sha256": digest,
                    "values": store.embedding_model.embed(item["text"]),
                }
            )
        )
        sources.append(
            {
                "record_id": item["item_id"],
                "source_text": {"status": "retained", "path": f"sources/{digest}.txt", "sha256": digest},
            }
        )
    document = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "stores": [
            {
                "store_id": store_id,
                "embedding_model": {
                    "id": PRIVATEAISTACK_HASHED_BOW,
                    "revision": EMBEDDING_REVISION,
                    "dimensions": store.embedding_model.dimensions,
                    "availability": "open_weight_self_hostable",
                    "availability_verification": "attested",
                },
                "vectors": {
                    "layout": "sharded_jsonl",
                    "shards": _shards(vector_lines, directory, "vectors", size),
                    "source_text": sources,
                },
            }
        ],
    }
    return _write_json(directory / "embedding-portability.json", document)


async def export_agent_memory(
    store: MemoryStore,
    out_dir: Path,
    *,
    store_id: str = DEFAULT_STORE_ID,
    export_id: str | None = None,
    redact_secrets: bool = False,
    exclude: Mapping[str, str] | None = None,
    include_embeddings: bool = True,
    shard_items: int = DEFAULT_SHARD_ITEMS,
    audit: AuditWriter | None = None,
    actor: str = "operator",
    now: datetime | None = None,
) -> ExportSummary:
    """Export the knowledge store as ``agent-memory.v1`` into an empty directory.

    Every chunk is checked before anything is written. ``exclude`` maps chunk
    IDs to the reason each is left out; an excluded chunk is not deleted, and
    each exclusion is written to ``audit`` as ``memory.export_exclusion``.
    """
    now = now or datetime.now(UTC)
    export_id = _identifier(export_id or f"exp-{now.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}", "export ID")
    _identifier(store_id, "store ID")
    if shard_items < 1:
        raise MemoryExportError("shard_items must be at least 1.")
    excluded = dict(exclude or {})
    if any(not str(reason).strip() for reason in excluded.values()):
        raise MemoryExportError("Every excluded chunk needs a reason.")
    if excluded and audit is None:
        raise MemoryExportError("Excluding chunks needs an audit log to record each exclusion in.")

    await store.initialize()
    chunks = sorted(await store.all_chunks(), key=lambda chunk: chunk.chunk_id)
    if len(chunks) > MAX_ITEMS:
        raise MemoryExportError(f"The store holds more than {MAX_ITEMS:,} chunks.")
    unknown = sorted(set(excluded) - {chunk.chunk_id for chunk in chunks})
    if unknown:
        raise MemoryExportError(f"Cannot exclude chunks the store does not hold: {', '.join(unknown)}.")
    kept = [chunk for chunk in chunks if chunk.chunk_id not in excluded]
    if include_embeddings:
        mismatched = _mismatched_vectors(store, kept)
        if mismatched:
            shown = ", ".join(mismatched[:5]) + (" ..." if len(mismatched) > 5 else "")
            raise MemoryExportError(
                f"{len(mismatched)} stored vector(s) are not this store's embedding of their chunk text "
                f"(dimension {store.embedding_model.dimensions}): {shown}. The index is inconsistent; "
                "re-ingest those documents, or export without embeddings."
            )
    items = [_item(chunk, redact_secrets) for chunk in kept]

    if out_dir.is_symlink() or (out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir()))):
        raise MemoryExportError("Export into an empty directory, not an existing one or a link.")
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = None
    if include_embeddings and items:
        inventory_sha = _embedding_inventory(store, items, out_dir, store_id, shard_items)
        embeddings = {
            "inventory": {"path": "embeddings/embedding-portability.json", "sha256": inventory_sha},
            "inventory_store_id": store_id,
        }
    layout: dict[str, Any]
    if items:
        layout = {"layout": "sharded_jsonl", "shards": _shards([_line(item) for item in items], out_dir, "items", shard_items)}
    else:
        layout = {"layout": "single_file", "records": []}
    document = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "generated_by": {"tool": "privateaistack", "version": __version__},
        "export": {
            "export_id": export_id,
            "created_at": _timestamp(now),
            "redaction_profile": "secrets" if redact_secrets else None,
            "source": {"system": SOURCE_SYSTEM, "backend": store.persistence_mode},
        },
        "stores": [
            {
                "store_id": store_id,
                "kind": "retrieval",
                "embeddings": embeddings,
                "items": layout,
                "counts": {"active": len(items), "quarantined": 0, "deleted_by_policy": 0},
                "tombstones": [],
            }
        ],
    }
    path = out_dir / DOCUMENT_NAME
    digest = _write_json(path, document)
    if audit is not None:
        for chunk_id, reason in sorted(excluded.items()):
            audit.write(
                EXCLUSION_EVENT,
                entity_type="memory_chunk",
                entity_id=chunk_id,
                actor=actor,
                details={"export_id": export_id, "export_sha256": digest, "reason": reason},
            )
    return ExportSummary(
        path=path,
        sha256=digest,
        export_id=export_id,
        items=len(items),
        embeddings=embeddings is not None,
        excluded=tuple(sorted(excluded)),
    )
