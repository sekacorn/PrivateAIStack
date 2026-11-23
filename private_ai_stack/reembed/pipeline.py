"""The offline re-embedding pipeline.

Checkpoint 3b-2. Given an ``embedding-portability.v1`` document with retained
source text, this rebuilds one of its stores with a different model and writes
three things into a target directory:

``vectors/shard-NNNN.jsonl``, ``sources/record-NNNN.txt``
    The rebuilt store's contents, one shard per batch.
``embedding-portability.json``
    The rebuilt store as an ordinary ``embedding-portability.v1`` document, so
    ``exit-report`` scores it with no new code.
``re-embed-record.json``
    A ``re-embed-record.v1`` saying what was rebuilt, what was skipped and why,
    how the run was batched, and what left the machine.

Nothing here opens a socket or loads weights. The target is one of the two
built-in hashing schemes, which are pure arithmetic over :mod:`hashlib`, so the
run is reproducible by anyone holding the same source text. Remote providers
arrive in 3b-5 behind an explicit opt-in, and the ``egress`` block is written
now so that a later remote run has nowhere to hide: a reader who sees the
block cannot mistake its absence for a local run.

The record this writes carries no ``quality`` block. A comparison has not been
run, and ``not_measured`` is a claim about a measurement that was attempted and
found insufficient. 3b-4 adds the comparison; until then the honest output is
silence.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from private_ai_stack.reembed.checkpoint import (
    CHECKPOINT_FILENAME,
    Checkpoint,
    CheckpointError,
    CompletedBatch,
    load_checkpoint,
    verify_resumable,
    write_bytes_atomically,
    write_checkpoint,
    write_json_atomically,
)
from private_ai_stack.reembed.embedders import Embedder, EmbedderError, ModelIdentity, build_built_in_embedder
from private_ai_stack.reembed.evidence import (
    DELETED_BY_POLICY,
    MISSING,
    SCHEMA_VERSION,
    EvidenceError,
    SourceStore,
    load_source_store,
)

DEFAULT_BATCH_SIZE: Final = 64
"""Bounded for a 16 GB machine, and bounded regardless of store size.

Sixty-four records at ``embedding-portability.v1``'s 1 MiB source-text ceiling
is 64 MiB of text, and 64 vectors at the 8192-dimension ceiling is under 5 MiB
as Python floats. The store is never held whole, so this holds for a store of
any size. Operators with more memory can raise it; the point of the default is
that nobody has to.
"""

MAX_BATCH_SIZE: Final = 10_000
MAX_SHARD_LINE_BYTES: Final = 64 * 1024
"""One JSONL line per record, matching what ``embedding-portability.v1`` streams.

A vector wide enough to exceed this would produce a document the format cannot
read back, so the run stops rather than writing one.
"""

MAX_SHARDS: Final = 1024
RECORD_SCHEMA_VERSION: Final = "re-embed-record.v1"
TARGET_DOCUMENT_FILENAME: Final = "embedding-portability.json"
RECORD_FILENAME: Final = "re-embed-record.json"
AVAILABILITY: Final = frozenset({"open_weight_self_hostable", "vendor_only"})
_SHARD_NAME: Final = re.compile(r"^shard-\d{4}\.jsonl$")


class ReEmbedError(ValueError):
    """Raised when a re-embedding run cannot proceed as declared."""


@dataclass(frozen=True)
class ReEmbedOutcome:
    """Where the run put things, and what it did."""

    target_document: Path
    record: Path
    checkpoint: Path
    total_records: int
    retained_rebuilt: int
    missing_skipped: int
    deleted_by_policy_skipped: int
    batches_completed: int
    batches_embedded_now: int
    resumed: bool

    @property
    def complete(self) -> bool:
        return self.retained_rebuilt + self.missing_skipped + self.deleted_by_policy_skipped == self.total_records


def _plan(store: SourceStore, batch_size: int) -> list[tuple[str, ...]]:
    """Split the retained records into the batches this run will write.

    Deterministic by construction: the records are already sorted by
    identifier, so the same input always yields the same batching, which is
    what lets a resume compare a checkpoint's claims against what this run
    would have done rather than against a count.
    """
    identifiers = [record.record_id for record in store.retained]
    return [tuple(identifiers[start : start + batch_size]) for start in range(0, len(identifiers), batch_size)]


def _shard_path(index: int) -> str:
    return f"vectors/shard-{index:04d}.jsonl"


def _source_path(position: int) -> str:
    """Name source copies by position, never by record identifier.

    A ``record_id`` may contain a colon, which is a valid identifier in the
    evidence format and an invalid filename on Windows, and in general letting
    operator-supplied text choose a filename is how path problems start.
    """
    return f"sources/record-{position:04d}.txt"


def _discard_unclaimed_shards(directory: Path, claimed: frozenset[str]) -> list[str]:
    """Delete shard files that no checkpoint entry claims.

    These are batches that were written but had not been recorded when the run
    stopped. Their vectors may be complete, partial or truncated, and nothing
    on disk says which, so they are discarded and re-embedded rather than
    trusted. Only files this pipeline names are ever removed.
    """
    vectors = directory / "vectors"
    if not vectors.is_dir():
        return []
    discarded = []
    for candidate in sorted(vectors.iterdir()):
        if not candidate.is_file() or not _SHARD_NAME.match(candidate.name):
            continue
        if f"vectors/{candidate.name}" not in claimed:
            candidate.unlink()
            discarded.append(candidate.name)
    return discarded


def _encode_shard(records: list[tuple[str, str, list[float]]]) -> bytes:
    """Encode one batch as JSONL, exactly as ``embedding-portability.v1`` streams it.

    Each line carries the digest of the source text the vector was produced
    from, not only the vector. That is what lets a reader check, later and
    offline, that the text beside a store is the text that store was built
    from; a line holding only values would make the store unverifiable.
    """
    lines = []
    for record_id, source_sha256, values in records:
        line = json.dumps({"record_id": record_id, "source_text_sha256": source_sha256, "values": values}, ensure_ascii=False)
        encoded = line.encode("utf-8")
        if len(encoded) > MAX_SHARD_LINE_BYTES:
            raise ReEmbedError(
                f"Vector for {record_id!r} encodes to {len(encoded)} bytes, over the {MAX_SHARD_LINE_BYTES} byte line limit."
            )
        lines.append(encoded)
    return b"\n".join(lines) + b"\n"


def _ensure_source_copy(directory: Path, position: int, text: str) -> tuple[str, str]:
    """Write one source copy into the target store, or keep an identical existing one.

    The rebuilt store carries its own copy of the text it was built from, which
    is what makes it a self-contained ``embedding-portability.v1`` input rather
    than one that only resolves next to its predecessor. The cost is a second
    copy of the retained text on disk, which is the honest price of the target
    store standing on its own.
    """
    relative = _source_path(position)
    path = directory / relative
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return relative, digest
    write_bytes_atomically(path, payload)
    return relative, digest


def _target_document(store_id: str, model: dict[str, Any], shards: list[dict[str, Any]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "stores": [
            {
                "store_id": store_id,
                "embedding_model": model,
                "vectors": {"layout": "sharded_jsonl", "shards": shards, "source_text": sources},
            }
        ],
    }


def _model_block(identity: ModelIdentity, availability: str, weights_sha256: str | None, license_name: str | None) -> dict[str, Any]:
    if availability not in AVAILABILITY:
        raise ReEmbedError(f"availability must be one of {sorted(AVAILABILITY)}, not {availability!r}.")
    model: dict[str, Any] = {
        "id": identity.id,
        "revision": identity.revision,
        "dimensions": identity.dimensions,
        "availability": availability,
        "availability_verification": "attested",
    }
    if weights_sha256 is not None:
        model["weights_sha256"] = weights_sha256
    if license_name is not None:
        model["license"] = license_name
    return model


def re_embed_store(
    *,
    source_document: Path,
    source_store_id: str,
    target_directory: Path,
    target_store_id: str,
    target_model: ModelIdentity,
    availability: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    weights_sha256: str | None = None,
    license_name: str | None = None,
    embedder: Embedder | None = None,
) -> ReEmbedOutcome:
    """Rebuild one store with ``target_model``, resuming an interrupted run if one is there.

    ``availability`` has no default on purpose. It is an operator attestation
    that no offline check can confirm, and defaulting it would put a claim into
    the evidence that nobody made.

    ``embedder`` exists so tests can supply a deliberately broken one; left
    unset, the built-in registry builds the embedder from ``target_model`` and
    an identity it does not implement is refused rather than approximated.
    """
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ReEmbedError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}, not {batch_size}.")
    try:
        store = load_source_store(Path(source_document), source_store_id)
    except EvidenceError as exc:
        raise ReEmbedError(str(exc)) from None
    try:
        engine = embedder if embedder is not None else build_built_in_embedder(target_model)
    except EmbedderError as exc:
        raise ReEmbedError(str(exc)) from None
    if engine.identity.triple != target_model.triple:
        raise ReEmbedError(f"Embedder identity {engine.identity.triple} does not match the declared target {target_model.triple}.")
    directory = Path(target_directory)
    directory.mkdir(parents=True, exist_ok=True)
    retained = store.retained
    plan = _plan(store, batch_size)
    if len(plan) > MAX_SHARDS:
        raise ReEmbedError(f"{len(plan)} batches exceeds the {MAX_SHARDS} shard limit; raise batch_size.")

    checkpoint_path = directory / CHECKPOINT_FILENAME
    checkpoint = Checkpoint(
        source_store_id=store.store_id,
        source_evidence_sha256=store.document_sha256,
        target_store_id=target_store_id,
        target_model=target_model,
        batch_size=batch_size,
    )
    resumed = False
    if checkpoint_path.exists():
        try:
            existing = load_checkpoint(checkpoint_path)
            verify_resumable(
                existing,
                source_store_id=store.store_id,
                source_evidence_sha256=store.document_sha256,
                target_store_id=target_store_id,
                target_model=target_model,
                batch_size=batch_size,
                expected_batches=plan,
                directory=directory,
            )
        except CheckpointError as exc:
            raise ReEmbedError(f"Cannot resume: {exc}") from None
        checkpoint = existing
        resumed = bool(existing.batches)
    _discard_unclaimed_shards(directory, checkpoint.claimed_shards)

    positions = {record.record_id: position for position, record in enumerate(retained)}
    by_id = {record.record_id: record for record in retained}
    sources: dict[str, tuple[str, str]] = {}
    embedded_now = 0

    for index, expected in enumerate(plan):
        if index < len(checkpoint.batches):
            continue
        produced: list[tuple[str, str, list[float]]] = []
        for record_id in expected:
            record = by_id[record_id]
            try:
                text = store.read_text(record)
            except EvidenceError as exc:
                raise ReEmbedError(str(exc)) from None
            _, source_digest = sources.setdefault(record_id, _ensure_source_copy(directory, positions[record_id], text))
            values = engine.embed(text)
            if len(values) != target_model.dimensions:
                raise ReEmbedError(
                    f"Embedder produced {len(values)} dimensions for {record_id!r}, but the declared model is "
                    f"{target_model.dimensions}-dimensional. Refusing to write the store."
                )
            produced.append((record_id, source_digest, values))
        relative = _shard_path(index)
        digest = write_bytes_atomically(directory / relative, _encode_shard(produced))
        checkpoint = checkpoint.with_batch(CompletedBatch(index=index, shard_path=relative, shard_sha256=digest, record_ids=expected))
        write_checkpoint(checkpoint_path, checkpoint)
        embedded_now += 1

    for record in retained:
        if record.record_id in sources:
            continue
        try:
            sources[record.record_id] = _ensure_source_copy(directory, positions[record.record_id], store.read_text(record))
        except EvidenceError as exc:
            raise ReEmbedError(str(exc)) from None

    model_block = _model_block(target_model, availability, weights_sha256, license_name)
    shards = [
        {"path": batch.shard_path, "sha256": batch.shard_sha256, "record_count": len(batch.record_ids)} for batch in checkpoint.batches
    ]
    source_entries = [
        {
            "record_id": record.record_id,
            "source_text": {"status": "retained", "path": sources[record.record_id][0], "sha256": sources[record.record_id][1]},
        }
        for record in retained
    ]
    if not shards or not source_entries:
        raise ReEmbedError(f"Store {source_store_id!r} has no retained records to rebuild.")
    target_document = directory / TARGET_DOCUMENT_FILENAME
    target_digest = write_json_atomically(target_document, _target_document(target_store_id, model_block, shards, source_entries))
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    missing = store.count(MISSING)
    deleted = store.count(DELETED_BY_POLICY)
    record_path = directory / RECORD_FILENAME
    write_json_atomically(
        record_path,
        {
            "schema_version": RECORD_SCHEMA_VERSION,
            "source_store": {"store_id": store.store_id, "store_evidence_sha256": store.document_sha256},
            "target_store": {"store_id": target_store_id, "store_evidence_sha256": target_digest},
            "target_model": model_block,
            "batch_log": {
                "batch_size": batch_size,
                "batches_completed": len(checkpoint.batches),
                "records_embedded": len(retained),
                "resumed": resumed,
                "checkpoint_sha256": checkpoint_digest,
            },
            "coverage": {
                "total_records": len(store.records),
                "retained_rebuilt": len(retained),
                "missing_skipped": missing,
                "deleted_by_policy_skipped": deleted,
                "failed": 0,
            },
            "egress": {"remote_provider_used": False, "records_transmitted": 0, "raw_source_text_transmitted": False},
        },
    )
    return ReEmbedOutcome(
        target_document=target_document,
        record=record_path,
        checkpoint=checkpoint_path,
        total_records=len(store.records),
        retained_rebuilt=len(retained),
        missing_skipped=missing,
        deleted_by_policy_skipped=deleted,
        batches_completed=len(checkpoint.batches),
        batches_embedded_now=embedded_now,
        resumed=resumed,
    )
