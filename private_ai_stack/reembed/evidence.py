"""Reading an ``embedding-portability.v1`` document as re-embedding input.

The document and its source files are operator-supplied and may have come from
anywhere, so every read here is bounded and every path is checked before it is
opened. The rules mirror the AI Infrastructure Suite's ``_evidence`` module,
which validates the same documents on the way out; they are restated rather
than imported because the two repositories deliberately take no dependency on
each other, and the contract between them is the document, not the code.

What this module does *not* do is load the store. It returns record
descriptors with paths and digests; the text is read one batch at a time by the
pipeline, so memory stays bounded no matter how large the store is.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

MAX_DOCUMENT_BYTES: Final = 4 * 1024 * 1024
MAX_SOURCE_TEXT_BYTES: Final = 1024 * 1024
MAX_RECORDS: Final = 100_000
MAX_PATH_CHARS: Final = 240
SCHEMA_VERSION: Final = "embedding-portability.v1"
RETAINED: Final = "retained"
MISSING: Final = "missing"
DELETED_BY_POLICY: Final = "deleted_by_policy"

_RELATIVE_PATH: Final = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]*(/[a-zA-Z0-9_][a-zA-Z0-9_.-]*)*$")
_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")


class EvidenceError(ValueError):
    """Raised when an evidence document, or a file it points at, cannot be trusted."""


def reject_unsafe_path(base: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``base``, refusing anything that could escape it.

    Absolute paths, traversal, drive letters, UNC prefixes and symbolic links
    or reparse points are all refused. A symlink is refused even when it
    happens to point somewhere harmless: the digest a reader checks is of the
    bytes at the path, and a link makes that a different question later.
    """
    if not relative or len(relative) > MAX_PATH_CHARS:
        raise EvidenceError(f"Path must be 1 to {MAX_PATH_CHARS} characters: {relative!r}")
    if not _RELATIVE_PATH.match(relative):
        raise EvidenceError(f"Path is not a portable project-relative path: {relative!r}")
    candidate = base / relative
    for parent in [candidate, *candidate.parents]:
        if parent == base.parent:
            break
        if parent.is_symlink():
            raise EvidenceError(f"Path crosses a symbolic link or reparse point: {relative!r}")
    resolved = Path(os.path.realpath(candidate))
    anchor = Path(os.path.realpath(base))
    if resolved != anchor and anchor not in resolved.parents:
        raise EvidenceError(f"Path resolves outside the evidence directory: {relative!r}")
    return candidate


def read_bounded_text(path: Path, limit: int) -> bytes:
    """Return the bytes at ``path``, refusing anything that is not a bounded regular file."""
    if path.is_symlink():
        raise EvidenceError(f"Refusing to read a symbolic link: {path.name}")
    try:
        stat = path.stat()
    except OSError as exc:
        raise EvidenceError(f"Cannot read {path.name}: {exc.strerror or 'unavailable'}") from None
    if not path.is_file():
        raise EvidenceError(f"Not a regular file: {path.name}")
    if stat.st_size > limit:
        raise EvidenceError(f"File exceeds the {limit} byte limit: {path.name}")
    return path.read_bytes()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise EvidenceError(f"Duplicate JSON key: {key!r}")
        seen[key] = value
    return seen


def _reject_constant(literal: str) -> float:
    raise EvidenceError(f"Non-finite JSON literal is not allowed: {literal}")


def load_json_document(data: bytes, label: str) -> dict[str, Any]:
    """Parse JSON strictly: no duplicate keys, no ``NaN`` or ``Infinity``, object at the root."""
    try:
        document = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except UnicodeDecodeError:
        raise EvidenceError(f"{label} is not valid UTF-8.") from None
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{label} is not valid JSON: {exc.msg} at line {exc.lineno}.") from None
    if not isinstance(document, dict):
        raise EvidenceError(f"{label} must be a JSON object.")
    return document


@dataclass(frozen=True)
class SourceRecord:
    """One record of the source store, without its text.

    ``relative_path`` and ``sha256`` are set only for ``retained`` records.
    ``vector_sha256`` carries the source digest recorded when the original
    vector was produced, when the layout states one per record.
    """

    record_id: str
    status: str
    relative_path: str | None = None
    sha256: str | None = None
    vector_sha256: str | None = None

    @property
    def is_retained(self) -> bool:
        return self.status == RETAINED


@dataclass(frozen=True)
class SourceStore:
    """A store selected out of an evidence document, ready to be rebuilt."""

    store_id: str
    model_id: str
    model_revision: str
    model_dimensions: int
    base_directory: Path
    document_sha256: str
    records: tuple[SourceRecord, ...]

    @property
    def retained(self) -> tuple[SourceRecord, ...]:
        """Retained records in a deterministic order, which is the order they are rebuilt in."""
        return tuple(sorted((record for record in self.records if record.is_retained), key=lambda item: item.record_id))

    def count(self, status: str) -> int:
        return sum(1 for record in self.records if record.status == status)

    def read_text(self, record: SourceRecord) -> str:
        """Read and verify one retained record's source text.

        The text is hashed against every digest the document declares for it.
        A mismatch stops the run: rebuilding from text that did not produce the
        original vectors would make the comparison that follows meaningless,
        and silently rebuilding from it would be worse than failing.
        """
        if not record.is_retained or record.relative_path is None or record.sha256 is None:
            raise EvidenceError(f"Record {record.record_id!r} has no retained source text.")
        path = reject_unsafe_path(self.base_directory, record.relative_path)
        raw = read_bounded_text(path, MAX_SOURCE_TEXT_BYTES)
        digest = hashlib.sha256(raw).hexdigest()
        for declared, where in ((record.sha256, "source_text.sha256"), (record.vector_sha256, "the vector record")):
            if declared is not None and digest != declared:
                raise EvidenceError(f"Source text for {record.record_id!r} hashes to {digest}, but {where} declares {declared}.")
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            raise EvidenceError(f"Source text for {record.record_id!r} is not valid UTF-8.") from None


def _require(document: dict[str, Any], key: str, kind: type, label: str) -> Any:
    value = document.get(key)
    if not isinstance(value, kind) or isinstance(value, bool) and kind is int:
        raise EvidenceError(f"{label} must have a {kind.__name__} {key!r}.")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value) or len(value) > 128:
        raise EvidenceError(f"{label} is not a valid identifier: {value!r}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.match(value):
        raise EvidenceError(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return value


def _source_text(entry: dict[str, Any], record_id: str) -> tuple[str, str | None, str | None]:
    block = entry.get("source_text")
    if not isinstance(block, dict):
        raise EvidenceError(f"Record {record_id!r} must have a source_text object.")
    status = block.get("status")
    if status == RETAINED:
        return (
            RETAINED,
            _require(block, "path", str, f"Record {record_id!r}"),
            _digest(block.get("sha256"), f"Record {record_id!r} source_text.sha256"),
        )
    if status in (MISSING, DELETED_BY_POLICY):
        return str(status), None, None
    raise EvidenceError(f"Record {record_id!r} has an unknown source_text status: {status!r}")


def _records(vectors: dict[str, Any]) -> Iterable[SourceRecord]:
    layout = vectors.get("layout")
    if layout == "single_file":
        entries = _require(vectors, "records", list, "A single_file store")
        for entry in entries:
            if not isinstance(entry, dict):
                raise EvidenceError("Every vector record must be a JSON object.")
            record_id = _identifier(entry.get("record_id"), "A vector record's record_id")
            status, path, sha256 = _source_text(entry, record_id)
            vector_digest = entry.get("source_text_sha256_in_vector_record")
            yield SourceRecord(
                record_id=record_id,
                status=status,
                relative_path=path,
                sha256=sha256,
                vector_sha256=_digest(vector_digest, f"Record {record_id!r}") if status == RETAINED else None,
            )
        return
    if layout == "sharded_jsonl":
        entries = _require(vectors, "source_text", list, "A sharded_jsonl store")
        for entry in entries:
            if not isinstance(entry, dict):
                raise EvidenceError("Every source_text entry must be a JSON object.")
            record_id = _identifier(entry.get("record_id"), "A source_text entry's record_id")
            status, path, sha256 = _source_text(entry, record_id)
            yield SourceRecord(record_id=record_id, status=status, relative_path=path, sha256=sha256)
        return
    raise EvidenceError(f"Unknown vector layout: {layout!r}")


def load_source_store(document_path: Path, store_id: str) -> SourceStore:
    """Load one store out of an ``embedding-portability.v1`` document.

    The returned store carries the digest of the document bytes, which is what
    the re-embed record binds to. An identifier selects the claim; the digest
    binds it to content.
    """
    path = Path(document_path)
    raw = read_bounded_text(path, MAX_DOCUMENT_BYTES)
    document = load_json_document(raw, f"{path.name}")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError(f"Expected {SCHEMA_VERSION}, found {document.get('schema_version')!r}.")
    stores = _require(document, "stores", list, "The document")
    matches = [store for store in stores if isinstance(store, dict) and store.get("store_id") == store_id]
    if not matches:
        found = ", ".join(sorted(str(store.get("store_id")) for store in stores if isinstance(store, dict)))
        raise EvidenceError(f"No store {store_id!r} in {path.name}. Found: {found or 'none'}.")
    if len(matches) > 1:
        raise EvidenceError(f"Store {store_id!r} appears {len(matches)} times in {path.name}.")
    store = matches[0]
    model = _require(store, "embedding_model", dict, f"Store {store_id!r}")
    vectors = _require(store, "vectors", dict, f"Store {store_id!r}")
    records = tuple(_records(vectors))
    if not records:
        raise EvidenceError(f"Store {store_id!r} has no records.")
    if len(records) > MAX_RECORDS:
        raise EvidenceError(f"Store {store_id!r} declares more than {MAX_RECORDS} records.")
    identifiers = [record.record_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise EvidenceError(f"Store {store_id!r} repeats a record_id.")
    dimensions = _require(model, "dimensions", int, f"Store {store_id!r} embedding_model")
    return SourceStore(
        store_id=_identifier(store_id, "store_id"),
        model_id=_require(model, "id", str, f"Store {store_id!r} embedding_model"),
        model_revision=_require(model, "revision", str, f"Store {store_id!r} embedding_model"),
        model_dimensions=dimensions,
        base_directory=path.parent,
        document_sha256=hashlib.sha256(raw).hexdigest(),
        records=records,
    )
