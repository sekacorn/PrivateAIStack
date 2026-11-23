"""The resume state of a re-embedding run: ``re-embed-checkpoint.v1``.

A run over a large store is long and interruptible, so it is a resumable batch
job rather than a single pass. The ordering rule is the whole of the crash
safety story:

1. the batch's shard is written and flushed to disk;
2. only then is the checkpoint updated to claim it, atomically.

So a crash can leave a shard that no checkpoint entry claims, but never a claim
without its shard. An unclaimed shard is an unfinished batch and is discarded
on resume rather than trusted; a claimed shard whose bytes no longer match its
recorded digest was changed by something other than this run, and that is an
error rather than a reason to quietly redo the work.

The schema lives beside this module in ``schemas/``. The checkpoint is run
state, not evidence: nothing downstream reads it, and ``exit-report`` never
sees it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from private_ai_stack.reembed.embedders import ModelIdentity
from private_ai_stack.reembed.evidence import EvidenceError, load_json_document, read_bounded_text

SCHEMA_VERSION: Final = "re-embed-checkpoint.v1"
MAX_CHECKPOINT_BYTES: Final = 8 * 1024 * 1024
CHECKPOINT_FILENAME: Final = "checkpoint.json"


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be resumed from as written."""


@dataclass(frozen=True)
class CompletedBatch:
    index: int
    shard_path: str
    shard_sha256: str
    record_ids: tuple[str, ...]


@dataclass(frozen=True)
class Checkpoint:
    """What a run has durably finished, bound to what it was finishing."""

    source_store_id: str
    source_evidence_sha256: str
    target_store_id: str
    target_model: ModelIdentity
    batch_size: int
    execution_fingerprint: str | None = None
    batches: tuple[CompletedBatch, ...] = ()

    @property
    def completed_record_ids(self) -> tuple[str, ...]:
        return tuple(record_id for batch in self.batches for record_id in batch.record_ids)

    @property
    def claimed_shards(self) -> frozenset[str]:
        return frozenset(batch.shard_path for batch in self.batches)

    def with_batch(self, batch: CompletedBatch) -> Checkpoint:
        return Checkpoint(
            source_store_id=self.source_store_id,
            source_evidence_sha256=self.source_evidence_sha256,
            target_store_id=self.target_store_id,
            target_model=self.target_model,
            batch_size=self.batch_size,
            execution_fingerprint=self.execution_fingerprint,
            batches=(*self.batches, batch),
        )

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source_store_id": self.source_store_id,
            "source_evidence_sha256": self.source_evidence_sha256,
            "target_store_id": self.target_store_id,
            "target_model": {
                "id": self.target_model.id,
                "revision": self.target_model.revision,
                "dimensions": self.target_model.dimensions,
            },
            "batch_size": self.batch_size,
            "batches": [
                {
                    "index": batch.index,
                    "shard_path": batch.shard_path,
                    "shard_sha256": batch.shard_sha256,
                    "record_ids": list(batch.record_ids),
                }
                for batch in self.batches
            ],
        }
        if self.execution_fingerprint is not None:
            document["execution_fingerprint"] = self.execution_fingerprint
        return document


def write_bytes_atomically(path: Path, payload: bytes) -> str:
    """Write ``payload`` so a reader sees either the old file or the whole new one.

    A temporary file in the same directory is written, flushed and fsynced,
    then renamed over the target. ``os.replace`` is atomic on both POSIX and
    Windows, which is why a truncated checkpoint means tampering or hardware
    trouble rather than an ordinary interruption. The returned digest is of the
    exact bytes written, so a caller never has to re-read what it just wrote.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def write_json_atomically(path: Path, document: dict[str, Any]) -> str:
    """Serialise ``document`` deterministically and write it atomically."""
    return write_bytes_atomically(path, (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def write_checkpoint(path: Path, checkpoint: Checkpoint) -> str:
    return write_json_atomically(path, checkpoint.to_document())


def _batch(entry: Any, position: int) -> CompletedBatch:
    if not isinstance(entry, dict):
        raise CheckpointError(f"Checkpoint batch {position} is not a JSON object.")
    index, shard_path, shard_sha256, record_ids = (
        entry.get("index"),
        entry.get("shard_path"),
        entry.get("shard_sha256"),
        entry.get("record_ids"),
    )
    if not isinstance(index, int) or isinstance(index, bool) or index != position:
        raise CheckpointError(f"Checkpoint batch at position {position} claims index {index!r}.")
    if not isinstance(shard_path, str) or not shard_path:
        raise CheckpointError(f"Checkpoint batch {position} has no shard_path.")
    if not isinstance(shard_sha256, str) or len(shard_sha256) != 64:
        raise CheckpointError(f"Checkpoint batch {position} has no valid shard_sha256.")
    if not isinstance(record_ids, list) or not record_ids or not all(isinstance(item, str) and item for item in record_ids):
        raise CheckpointError(f"Checkpoint batch {position} has no valid record_ids.")
    return CompletedBatch(index=index, shard_path=shard_path, shard_sha256=shard_sha256, record_ids=tuple(record_ids))


def load_checkpoint(path: Path) -> Checkpoint:
    """Read a checkpoint, refusing anything structurally unusable.

    This checks the document against itself only. Whether it matches the run
    being resumed is :func:`verify_resumable`'s job, so that a mismatch can
    name which of the two is wrong.
    """
    try:
        raw = read_bounded_text(path, MAX_CHECKPOINT_BYTES)
        document = load_json_document(raw, CHECKPOINT_FILENAME)
    except EvidenceError as exc:
        raise CheckpointError(str(exc)) from None
    if document.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError(f"Expected {SCHEMA_VERSION}, found {document.get('schema_version')!r}.")
    model = document.get("target_model")
    if not isinstance(model, dict):
        raise CheckpointError("Checkpoint has no target_model object.")
    batch_size = document.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise CheckpointError(f"Checkpoint batch_size is not a positive integer: {batch_size!r}")
    entries = document.get("batches")
    if not isinstance(entries, list):
        raise CheckpointError("Checkpoint has no batches array.")
    batches = tuple(_batch(entry, position) for position, entry in enumerate(entries))
    identifiers = [record_id for batch in batches for record_id in batch.record_ids]
    if len(set(identifiers)) != len(identifiers):
        raise CheckpointError("Checkpoint claims the same record in more than one batch.")
    try:
        identity = ModelIdentity(id=str(model.get("id")), revision=str(model.get("revision")), dimensions=int(model.get("dimensions", 0)))
    except (ValueError, TypeError) as exc:
        raise CheckpointError(f"Checkpoint target_model is not a valid identity: {exc}") from None
    for key in ("source_store_id", "source_evidence_sha256", "target_store_id"):
        if not isinstance(document.get(key), str) or not document.get(key):
            raise CheckpointError(f"Checkpoint has no {key}.")
    execution_fingerprint = document.get("execution_fingerprint")
    if execution_fingerprint is not None and (
        not isinstance(execution_fingerprint, str)
        or len(execution_fingerprint) != 64
        or any(char not in "0123456789abcdef" for char in execution_fingerprint)
    ):
        raise CheckpointError("Checkpoint has no valid execution_fingerprint.")
    return Checkpoint(
        source_store_id=str(document["source_store_id"]),
        source_evidence_sha256=str(document["source_evidence_sha256"]),
        target_store_id=str(document["target_store_id"]),
        target_model=identity,
        batch_size=batch_size,
        execution_fingerprint=execution_fingerprint,
        batches=batches,
    )


def verify_resumable(
    checkpoint: Checkpoint,
    *,
    source_store_id: str,
    source_evidence_sha256: str,
    target_store_id: str,
    target_model: ModelIdentity,
    batch_size: int,
    execution_fingerprint: str,
    require_execution_binding: bool,
    expected_batches: list[tuple[str, ...]],
    directory: Path,
) -> None:
    """Refuse to resume a checkpoint that does not describe this exact run.

    Every check here exists because its absence would corrupt a store quietly.
    A different input document, a different target, a different model triple or
    a different batch size all change which vectors belong where. So does a
    reordered or edited input: each completed batch must hold exactly the
    records this run would have put in that batch, in that order, which is why
    the expected batching is recomputed and compared rather than trusted.
    """
    if checkpoint.source_evidence_sha256 != source_evidence_sha256:
        raise CheckpointError(
            "Checkpoint was written against a different input document "
            f"({checkpoint.source_evidence_sha256[:12]}..., now {source_evidence_sha256[:12]}...). Refusing to resume."
        )
    if checkpoint.source_store_id != source_store_id:
        raise CheckpointError(f"Checkpoint is for source store {checkpoint.source_store_id!r}, not {source_store_id!r}.")
    if checkpoint.target_store_id != target_store_id:
        raise CheckpointError(f"Checkpoint is for target store {checkpoint.target_store_id!r}, not {target_store_id!r}.")
    if checkpoint.target_model.triple != target_model.triple:
        raise CheckpointError(f"Checkpoint is for model {checkpoint.target_model.triple}, not {target_model.triple}.")
    if checkpoint.batch_size != batch_size:
        raise CheckpointError(f"Checkpoint used batch size {checkpoint.batch_size}, not {batch_size}.")
    if checkpoint.execution_fingerprint is None:
        if require_execution_binding:
            raise CheckpointError("An external execution cannot resume from a checkpoint without an execution binding.")
    elif checkpoint.execution_fingerprint != execution_fingerprint:
        raise CheckpointError("Checkpoint was written for a different executor or consent context.")
    if len(checkpoint.batches) > len(expected_batches):
        raise CheckpointError(f"Checkpoint claims {len(checkpoint.batches)} batches, but this run has {len(expected_batches)}.")
    for batch in checkpoint.batches:
        expected = expected_batches[batch.index]
        if batch.record_ids != expected:
            raise CheckpointError(
                f"Checkpoint batch {batch.index} claims records that this run would not have placed there. Refusing to resume."
            )
        shard = directory / batch.shard_path
        if not shard.is_file():
            raise CheckpointError(f"Checkpoint claims shard {batch.shard_path!r}, which is not on disk.")
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        if digest != batch.shard_sha256:
            raise CheckpointError(
                f"Shard {batch.shard_path!r} hashes to {digest[:12]}..., but the checkpoint claims {batch.shard_sha256[:12]}.... "
                "Something other than this run changed it."
            )
