"""Builders for ``embedding-portability.v1`` input used by the re-embedding tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

RETAINED = "retained"


def write_document(directory: Path, document: dict[str, Any], name: str = "embedding-portability.json") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def build_source_store(
    directory: Path,
    *,
    store_id: str = "support-memory",
    texts: dict[str, str] | None = None,
    missing: tuple[str, ...] = (),
    deleted_by_policy: tuple[str, ...] = (),
    dimensions: int = 8,
    layout: str = "single_file",
) -> Path:
    """Write a source evidence document with retained text on disk.

    ``texts`` maps record identifiers to their source text; ``missing`` and
    ``deleted_by_policy`` add records with those statuses and no text at all,
    which is what the pipeline must refuse to send to any model.
    """
    texts = (
        texts
        if texts is not None
        else {
            "procedure-escalation": "Escalate to the on-call engineer when the queue depth exceeds 40.",
            "procedure-password-reset": "Reset the password through the self-service portal.",
            "procedure-refund": "Refunds over 500 need a supervisor approval and an audit entry.",
        }
    )
    sources = directory / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    retained_entries = []
    for record_id, text in texts.items():
        payload = text.encode("utf-8")
        (sources / f"{record_id}.txt").write_bytes(payload)
        retained_entries.append((record_id, f"sources/{record_id}.txt", hashlib.sha256(payload).hexdigest()))

    if layout == "single_file":
        records: list[dict[str, Any]] = [
            {
                "record_id": record_id,
                "vector": {"path": f"vectors/{record_id}.json", "sha256": "0" * 64},
                "source_text_sha256_in_vector_record": digest,
                "source_text": {"status": RETAINED, "path": path, "sha256": digest},
            }
            for record_id, path, digest in retained_entries
        ]
        records.extend(
            {
                "record_id": record_id,
                "vector": {"path": f"vectors/{record_id}.json", "sha256": "0" * 64},
                "source_text_sha256_in_vector_record": "0" * 64,
                "source_text": {"status": status},
            }
            for status, group in (("missing", missing), ("deleted_by_policy", deleted_by_policy))
            for record_id in group
        )
        vectors: dict[str, Any] = {"layout": "single_file", "records": records}
    else:
        entries: list[dict[str, Any]] = [
            {"record_id": record_id, "source_text": {"status": RETAINED, "path": path, "sha256": digest}}
            for record_id, path, digest in retained_entries
        ]
        entries.extend(
            {"record_id": record_id, "source_text": {"status": status}}
            for status, group in (("missing", missing), ("deleted_by_policy", deleted_by_policy))
            for record_id in group
        )
        vectors = {
            "layout": "sharded_jsonl",
            "shards": [{"path": "vectors/shard-0000.jsonl", "sha256": "0" * 64, "record_count": len(entries)}],
            "source_text": entries,
        }

    document = {
        "schema_version": "embedding-portability.v1",
        "stores": [
            {
                "store_id": store_id,
                "embedding_model": {
                    "id": "example-embed-small",
                    "revision": "2026-02-01",
                    "dimensions": dimensions,
                    "availability": "open_weight_self_hostable",
                    "availability_verification": "attested",
                },
                "vectors": vectors,
            }
        ],
    }
    return write_document(directory, document)


def build_ranked_store(
    directory: Path,
    *,
    store_id: str = "old-store",
    model_id: str = "agentforge.hashed-bow.blake2b",
    revision: str = "1",
    dimensions: int = 64,
    count: int = 40,
    text: Callable[[int], str] | None = None,
) -> Path:
    """Write a store whose vectors are real, so a quality comparison can rank it.

    :func:`build_source_store` writes placeholder vector digests because the
    pipeline only ever reads source text. The comparison reads the vectors
    themselves and checks every digest, so it needs a store that was actually
    embedded.
    """
    from private_ai_stack.reembed.embedders import ModelIdentity, build_built_in_embedder

    embedder = build_built_in_embedder(ModelIdentity(id=model_id, revision=revision, dimensions=dimensions))
    body = text if text is not None else (lambda index: f"record {index} about topic {index % 7} and shared vocabulary")
    (directory / "sources").mkdir(parents=True, exist_ok=True)
    (directory / "vectors").mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    for index in range(count):
        record_id = f"r{index:04d}"
        payload = body(index).encode("utf-8")
        relative = f"sources/record-{index:04d}.txt"
        (directory / relative).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        entries.append({"record_id": record_id, "source_text": {"status": RETAINED, "path": relative, "sha256": digest}})
        values = embedder.embed(payload.decode("utf-8"))
        lines.append(json.dumps({"record_id": record_id, "source_text_sha256": digest, "values": values}))
    shard = ("\n".join(lines) + "\n").encode("utf-8")
    (directory / "vectors/shard-0000.jsonl").write_bytes(shard)
    document = {
        "schema_version": "embedding-portability.v1",
        "stores": [
            {
                "store_id": store_id,
                "embedding_model": {
                    "id": model_id,
                    "revision": revision,
                    "dimensions": dimensions,
                    "availability": "open_weight_self_hostable",
                    "availability_verification": "attested",
                },
                "vectors": {
                    "layout": "sharded_jsonl",
                    "shards": [
                        {
                            "path": "vectors/shard-0000.jsonl",
                            "sha256": hashlib.sha256(shard).hexdigest(),
                            "record_count": count,
                        }
                    ],
                    "source_text": entries,
                },
            }
        ],
    }
    return write_document(directory, document)


def tree_digest(directory: Path, *, skip: tuple[str, ...] = ()) -> dict[str, str]:
    """Digest every file under ``directory`` by relative path, for byte-identity checks."""
    digests = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in skip:
            continue
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests
