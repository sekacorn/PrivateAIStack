"""Comparing retrieval behaviour before and after a re-embedding run.

Checkpoint 3b-4. Rebuilding a store answers "can this be rebuilt?". This
answers the question that follows: *does the rebuilt store still put the same
things near each other?* Everything here is arithmetic over vectors already on
disk. Nothing opens a socket or loads weights.

What this measures, and what it does not
----------------------------------------

The default mode is ``record_anchored``: a retained record's own stored vector
is the query in each store. That is not a convenience. Ranking the *old* store
for a free-text query means embedding that query with the *old* model, which at
a lock-in score of 50 is exactly the model the operator no longer has. An
anchor needs no model at query time, so the comparison works in the case the
whole phase exists for.

The price is that the result is **neighborhood agreement**, never retrieval
relevance. It says the two stores place the same records near each other. It
says nothing about whether a person searching either store finds what they
wanted, and nothing built on this result may describe it that way. The record
carries ``metric_basis`` so a reader does not have to infer it, and the AI
Infrastructure Suite refuses a record-anchored result that claims otherwise.

``text`` mode embeds free-text queries with each store's own model and is
available only when both identities are built-in schemes this package
implements, because that is the only case where the old model can be re-run
offline. With expected records it measures relevance; without them it is
agreement again.

Who chose the queries
---------------------

``seeded_random`` anchors are drawn by the same seeded hash the cost estimate
uses, so another party redraws the same set from the seed in the record. Only
those may move a lock-in score. An ``operator`` set is reported in full --
label, metrics and checks -- and moves nothing, because an operator who may
choose the queries may choose the result, and no offline check can tell a fair
hand-picked set from a flattering one.

Thresholds
----------

There are no built-in agreement floors. An operator says what acceptable means
before asking whether it was met; a tool that supplied the bar would be grading
its own homework. The declared bar travels with the verdict in the record's
``thresholds``, so no reader sees "sampled" without seeing what was cleared.

A run with no agreement floor still gets its numbers. It is reported as
``unjudged``: the metrics are real and the verdict does not exist, the result
names the floors that would produce one, and it can never move a lock-in score.
Throwing the measurement away because a declaration was missing would be the
wrong trade in the other direction.
"""

from __future__ import annotations

import hashlib
import json
import math
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from private_ai_stack.reembed.embedders import (
    BUILT_IN_MODEL_IDS,
    Embedder,
    EmbedderError,
    ModelIdentity,
    build_built_in_embedder,
)
from private_ai_stack.reembed.evidence import (
    SCHEMA_VERSION as PORTABILITY_SCHEMA,
)
from private_ai_stack.reembed.evidence import (
    EvidenceError,
    load_json_document,
    read_bounded_text,
    reject_unsafe_path,
)

QUERY_SET_SCHEMA_VERSION: Final = "golden-query-set.v1"
"""The operator-supplied query set this module reads. Versioned like every other format here."""

RECORD_ANCHORED: Final = "record_anchored"
TEXT: Final = "text"
OPERATOR: Final = "operator"
SEEDED_RANDOM: Final = "seeded_random"
NEIGHBORHOOD_AGREEMENT: Final = "neighborhood_agreement"
RETRIEVAL_RELEVANCE: Final = "retrieval_relevance"

SAMPLED: Final = "sampled"
BELOW_THRESHOLD: Final = "below_threshold"
UNJUDGED: Final = "unjudged"
NOT_MEASURED: Final = "not_measured"

SIMILARITY: Final = "cosine"
TIE_BREAK: Final = "ascending record_id among equal similarities"

DEFAULT_TOP_K: Final = 10
DEFAULT_MINIMUM_SAMPLE_SIZE: Final = 20
"""Matches ModelSwapBench's ``GateThresholds.minimum_sample_size``.

The two projects take no dependency on each other, so the number is restated
rather than imported. It is the same number on purpose: an operator who has
calibrated one should not have to calibrate the other.
"""

DEFAULT_ANCHOR_COUNT: Final = 32
DEFAULT_CANDIDATE_POOL: Final = 256
DEFAULT_SEED: Final = "re-embed-quality"
"""Retained only as the fallback when no store digests are available.

A seed an operator can choose is a seed an operator can grind: draw query set
after query set until the numbers clear the bar, then publish the one that
passed. :func:`derive_anchor_seed` is what a real run uses, and the AI
Infrastructure Suite treats any other seed as an operator-chosen query set --
reported in full, moving no lock-in score.
"""


def derive_anchor_seed(source_store_sha256: str, target_store_sha256: str) -> str:
    """The only anchor seed a record may carry and still move a score.

    A function of the two store digests the record already commits to, so
    neither the operator nor this tool can steer it, and a third party
    recomputes the same query set from the record alone. The suite derives the
    same value independently; if these two ever disagree, every record this
    tool writes silently stops being able to move a score, which is why the
    formula is pinned by a test on both sides.
    """
    payload = f"re-embed-anchor:{source_store_sha256}:{target_store_sha256}"
    return hashlib.sha256(payload.encode()).hexdigest()


MAX_TOP_K: Final = 1000
MAX_ANCHORS: Final = 1000
MAX_CANDIDATE_POOL: Final = 20_000
MAX_VECTOR_FLOATS: Final = 8_000_000
"""Eight million doubles is 64 MiB per store as :class:`array.array`, 128 MiB for both.

Vectors are held as packed doubles rather than Python lists because a list of
eight million floats is closer to a quarter of a gigabyte, and this has to run
on a 16 GB machine beside everything else.
"""

MAX_SIMILARITY_OPERATIONS: Final = 200_000_000
MAX_QUERY_SET_BYTES: Final = 1024 * 1024
MAX_QUERY_TEXT_CHARS: Final = 4096
MAX_EXPECTED_RECORDS: Final = 64


class QualityError(ValueError):
    """Raised when a quality comparison cannot be run or reported as declared."""


@dataclass(frozen=True)
class QualityThresholds:
    """What this operator counts as acceptable.

    Every agreement floor defaults to ``None``. A floor this package invented
    would be a bar nobody set, and a report showing a verdict against it would
    be showing the tool's opinion in the operator's voice. The drop caps have
    defaults because they bound a *change* against a measured baseline rather
    than declaring an absolute standard, and they only ever produce a check in
    ``text`` mode with expected records.
    """

    top_k: int = DEFAULT_TOP_K
    minimum_sample_size: int = DEFAULT_MINIMUM_SAMPLE_SIZE
    minimum_top_k_overlap: float | None = None
    minimum_rank_correlation: float | None = None
    maximum_hit_rate_drop: float | None = 0.05
    maximum_mrr_drop: float | None = 0.05

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= MAX_TOP_K:
            raise QualityError(f"top_k must be between 1 and {MAX_TOP_K}, not {self.top_k}.")
        if self.minimum_sample_size < 1:
            raise QualityError("minimum_sample_size must be at least 1.")
        for name, low, high in (
            ("minimum_top_k_overlap", 0.0, 1.0),
            ("minimum_rank_correlation", -1.0, 1.0),
            ("maximum_hit_rate_drop", 0.0, 1.0),
            ("maximum_mrr_drop", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise QualityError(f"{name} must be a finite number or unset.")
            if not low <= float(value) <= high:
                raise QualityError(f"{name} must be between {low} and {high}, not {value}.")


@dataclass(frozen=True)
class Check:
    """One threshold and what was measured against it."""

    name: str
    passed: bool
    actual: float
    threshold: float
    detail: str | None = None

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "name": self.name,
            "passed": self.passed,
            "actual": round(self.actual, 6),
            "threshold": self.threshold,
        }
        if self.detail is not None:
            document["detail"] = self.detail
        return document


@dataclass(frozen=True)
class QuerySet:
    """The queries a comparison will run, and the record of where they came from."""

    mode: str
    selection: str
    seed: str | None
    anchors: tuple[str, ...] = ()
    queries: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """``(text, expected_record_ids)`` pairs, for ``text`` mode only."""

    @property
    def declared_size(self) -> int:
        return len(self.anchors) if self.mode == RECORD_ANCHORED else len(self.queries)

    @property
    def has_expected_records(self) -> bool:
        return any(expected for _, expected in self.queries)


@dataclass(frozen=True)
class QualityResult:
    """What the comparison found, ready to be written into a re-embed record."""

    status: str
    mode: str
    metric_basis: str
    anchor_selection: str
    anchor_seed: str | None
    query_set_size: int
    top_k: int
    candidate_pool_size: int
    sufficient_evidence: bool
    metrics: dict[str, float]
    checks: tuple[Check, ...] = ()
    thresholds: dict[str, float] = field(default_factory=dict)
    boundary_ties: int = 0
    queries_skipped: int = 0
    correlation_undefined: bool = False
    top_k_capped: bool = False
    notes: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == SAMPLED

    @property
    def judged(self) -> bool:
        """Whether a verdict was reached at all.

        ``unjudged`` means the comparison ran, produced metrics, and was held
        against no agreement floor because none was declared. The numbers are
        real and the verdict does not exist, so nothing may read it as a pass.
        """
        return self.status in (SAMPLED, BELOW_THRESHOLD)

    @property
    def measures_user_search_quality(self) -> bool:
        """Always false for neighborhood agreement. Here to be checked, not assumed."""
        return self.metric_basis == RETRIEVAL_RELEVANCE

    def threshold_summary(self) -> str | None:
        """The declared bar as one line, to print beside the verdict."""
        if not self.thresholds:
            return None
        parts = ", ".join(f"{name} {value:g}" for name, value in sorted(self.thresholds.items()))
        return f"operator-declared thresholds: {parts}"

    def to_document(self) -> dict[str, Any]:
        """The ``quality`` block of a ``re-embed-record.v1``.

        Reading the fields beyond the original three needs
        ai-infrastructure-suite 0.1.0b3 or later; they are additive, so the
        record stays ``re-embed-record.v1``.
        """
        document: dict[str, Any] = {
            "status": self.status,
            "query_set_size": self.query_set_size,
            "top_k": self.top_k,
            "sufficient_evidence": self.sufficient_evidence,
            "mode": self.mode,
            "metric_basis": self.metric_basis,
            "anchor_selection": self.anchor_selection,
            "similarity": SIMILARITY,
            "candidate_pool_size": self.candidate_pool_size,
            "tie_break": TIE_BREAK,
            "boundary_ties": self.boundary_ties,
            "queries_skipped": self.queries_skipped,
            "correlation_undefined": self.correlation_undefined,
            "top_k_capped": self.top_k_capped,
        }
        if self.anchor_seed is not None:
            document["anchor_seed"] = self.anchor_seed
        if self.status != NOT_MEASURED:
            document["metrics"] = {name: round(value, 6) for name, value in sorted(self.metrics.items())}
            document["checks"] = [check.to_document() for check in self.checks]
            document["thresholds"] = dict(sorted(self.thresholds.items()))
        if self.notes is not None:
            document["notes"] = self.notes[:512]
        return document


# --- loading vectors --------------------------------------------------------


def _values(raw: object, dimensions: int, label: str) -> array[float]:
    if not isinstance(raw, list) or len(raw) != dimensions:
        raise QualityError(f"{label} must hold exactly {dimensions} values.")
    packed = array("d")
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise QualityError(f"{label} must hold finite numbers.")
        packed.append(float(item))
    return packed


def _check_digest(data: bytes, declared: object, label: str) -> None:
    if not isinstance(declared, str) or hashlib.sha256(data).hexdigest() != declared:
        raise QualityError(f"{label} differs from the digest the document declares for it.")


@dataclass(frozen=True)
class VectorStore:
    """One store's vectors, in memory, with the identity that produced them."""

    store_id: str
    model: ModelIdentity
    vectors: dict[str, array[float]]

    @property
    def record_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.vectors))


def _model_identity(block: dict[str, Any], store_id: str) -> ModelIdentity:
    try:
        return ModelIdentity(
            id=str(block.get("id")),
            revision=str(block.get("revision")),
            dimensions=int(block["dimensions"]),
        )
    except (EmbedderError, KeyError, TypeError, ValueError) as exc:
        raise QualityError(f"Store {store_id!r} does not declare a usable model identity: {exc}") from None


def _budget(count: int, dimensions: int, store_id: str) -> None:
    if count * dimensions > MAX_VECTOR_FLOATS:
        raise QualityError(
            f"Store {store_id!r} holds {count} vectors of {dimensions} dimensions, over the "
            f"{MAX_VECTOR_FLOATS} value budget this comparison keeps so it stays within memory."
        )


def load_vector_store(document_path: Path, store_id: str) -> VectorStore:
    """Read one store's vectors out of an ``embedding-portability.v1`` document.

    Every shard and vector file is checked against the digest the document
    declares for it. A store whose bytes have drifted from its own manifest is
    not a store this comparison will rank: the whole point is to say something
    about the vectors a reader can re-derive.
    """
    path = Path(document_path)
    raw = read_bounded_text(path, 4 * 1024 * 1024)
    try:
        document = load_json_document(raw, path.name)
    except EvidenceError as exc:
        raise QualityError(str(exc)) from None
    if document.get("schema_version") != PORTABILITY_SCHEMA:
        raise QualityError(f"Expected {PORTABILITY_SCHEMA}, found {document.get('schema_version')!r}.")
    stores = document.get("stores")
    if not isinstance(stores, list):
        raise QualityError(f"{path.name} has no stores array.")
    matches = [item for item in stores if isinstance(item, dict) and item.get("store_id") == store_id]
    if len(matches) != 1:
        raise QualityError(f"Expected exactly one store {store_id!r} in {path.name}, found {len(matches)}.")
    store = matches[0]
    model_block = store.get("embedding_model")
    vectors_block = store.get("vectors")
    if not isinstance(model_block, dict) or not isinstance(vectors_block, dict):
        raise QualityError(f"Store {store_id!r} is missing its model or vectors.")
    model = _model_identity(model_block, store_id)
    base = path.parent

    vectors: dict[str, array[float]] = {}
    layout = vectors_block.get("layout")
    if layout == "single_file":
        records = vectors_block.get("records")
        if not isinstance(records, list):
            raise QualityError(f"Store {store_id!r} declares no vector records.")
        _budget(len(records), model.dimensions, store_id)
        for entry in records:
            if not isinstance(entry, dict):
                raise QualityError("Every vector record must be a JSON object.")
            record_id = str(entry.get("record_id"))
            reference = entry.get("vector")
            if not isinstance(reference, dict):
                raise QualityError(f"Record {record_id!r} has no vector reference.")
            try:
                file_path = reject_unsafe_path(base, str(reference.get("path")))
                data = read_bounded_text(file_path, MAX_VECTOR_FLOATS)
            except EvidenceError as exc:
                raise QualityError(str(exc)) from None
            _check_digest(data, reference.get("sha256"), f"Vector file for {record_id!r}")
            try:
                content = load_json_document(data, f"vector file for {record_id!r}")
            except EvidenceError as exc:
                raise QualityError(str(exc)) from None
            vectors[record_id] = _values(content.get("values"), model.dimensions, f"Vector for {record_id!r}")
    elif layout == "sharded_jsonl":
        shards = vectors_block.get("shards")
        if not isinstance(shards, list):
            raise QualityError(f"Store {store_id!r} declares no shards.")
        declared = sum(entry.get("record_count", 0) for entry in shards if isinstance(entry, dict))
        _budget(int(declared), model.dimensions, store_id)
        for entry in shards:
            if not isinstance(entry, dict):
                raise QualityError("Every shard entry must be a JSON object.")
            try:
                shard_path = reject_unsafe_path(base, str(entry.get("path")))
                data = read_bounded_text(shard_path, MAX_VECTOR_FLOATS)
            except EvidenceError as exc:
                raise QualityError(str(exc)) from None
            _check_digest(data, entry.get("sha256"), f"Shard {entry.get('path')!r}")
            for line in data.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = load_json_document(line.encode("utf-8"), "shard line")
                except EvidenceError as exc:
                    raise QualityError(str(exc)) from None
                record_id = str(row.get("record_id"))
                vectors[record_id] = _values(row.get("values"), model.dimensions, f"Vector for {record_id!r}")
    else:
        raise QualityError(f"Unknown vector layout: {layout!r}")

    if not vectors:
        raise QualityError(f"Store {store_id!r} holds no vectors to compare.")
    return VectorStore(store_id=store_id, model=model, vectors=vectors)


# --- query sets -------------------------------------------------------------


def _identifier_list(raw: object, label: str, limit: int) -> tuple[str, ...]:
    if not isinstance(raw, list) or len(raw) > limit:
        raise QualityError(f"{label} must be a list of at most {limit} record identifiers.")
    values = []
    for item in raw:
        if not isinstance(item, str) or not item or len(item) > 128:
            raise QualityError(f"{label} must hold record identifiers.")
        values.append(item)
    if len(set(values)) != len(values):
        raise QualityError(f"{label} repeats a record identifier.")
    return tuple(values)


def load_query_set(path: Path) -> QuerySet:
    """Read a ``golden-query-set.v1`` document.

    A set read from a file is always ``operator`` selection, whatever it says
    about itself, because the file is what the operator chose to hand over.
    """
    try:
        raw = read_bounded_text(Path(path), MAX_QUERY_SET_BYTES)
        document = load_json_document(raw, Path(path).name)
    except EvidenceError as exc:
        raise QualityError(str(exc)) from None
    if document.get("schema_version") != QUERY_SET_SCHEMA_VERSION:
        raise QualityError(f"Expected {QUERY_SET_SCHEMA_VERSION}, found {document.get('schema_version')!r}.")
    mode = document.get("mode")
    if mode == RECORD_ANCHORED:
        anchors = _identifier_list(document.get("anchors"), "anchors", MAX_ANCHORS)
        if "queries" in document:
            raise QualityError("A record_anchored query set carries anchors, not queries.")
        return QuerySet(mode=RECORD_ANCHORED, selection=OPERATOR, seed=None, anchors=anchors)
    if mode == TEXT:
        entries = document.get("queries")
        if not isinstance(entries, list) or len(entries) > MAX_ANCHORS:
            raise QualityError(f"queries must be a list of at most {MAX_ANCHORS} entries.")
        if "anchors" in document:
            raise QualityError("A text query set carries queries, not anchors.")
        queries = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise QualityError("Every query must be a JSON object.")
            text = entry.get("text")
            if not isinstance(text, str) or not text or len(text) > MAX_QUERY_TEXT_CHARS:
                raise QualityError(f"Every query needs text of 1 to {MAX_QUERY_TEXT_CHARS} characters.")
            expected = _identifier_list(entry.get("expected_records", []), "expected_records", MAX_EXPECTED_RECORDS)
            queries.append((text, expected))
        return QuerySet(mode=TEXT, selection=OPERATOR, seed=None, queries=tuple(queries))
    raise QualityError(f"Unknown query set mode: {mode!r}")


def draw_anchors(record_ids: tuple[str, ...], *, count: int, seed: str, derived: bool = True) -> QuerySet:
    """Draw a reproducible anchor set by seeded hash over the record identifiers.

    The same ordering the cost estimate uses, for the same reason: taking the
    first N identifiers would be a sample of whatever the identifiers happen to
    correlate with, and nothing says they are unrelated to record size or age.

    ``derived`` says where the seed came from, and it is the whole difference
    between a draw and a choice. A seed from :func:`derive_anchor_seed` is a
    function of content the record commits to, so nobody could have ground it,
    and the set is reported as ``seeded_random``. A seed somebody picked is
    reported as ``operator`` even though the draw itself is mechanical, because
    an operator who can pick the seed can keep picking until the numbers pass.
    """
    if not 0 <= count <= MAX_ANCHORS:
        raise QualityError(f"The anchor count must be between 0 and {MAX_ANCHORS}, not {count}.")
    if not seed or len(seed) > 128:
        raise QualityError("The anchor seed must be 1 to 128 characters.")
    ordered = sorted(record_ids, key=lambda record_id: hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest())
    return QuerySet(
        mode=RECORD_ANCHORED,
        selection=SEEDED_RANDOM if derived else OPERATOR,
        seed=seed if derived else None,
        anchors=tuple(ordered[:count]),
    )


def _draw_pool(record_ids: tuple[str, ...], *, size: int, seed: str, keep: set[str]) -> tuple[str, ...]:
    """Pick the candidates every query is ranked against.

    Every anchor is kept whatever the draw says, so an anchor can never be
    missing from the pool it is ranked inside. The rest is the same seeded
    ordering, which makes the pool redrawable from the record alone.
    """
    if size >= len(record_ids):
        return tuple(sorted(record_ids))
    ordered = sorted(record_ids, key=lambda record_id: hashlib.sha256(f"{seed}:pool:{record_id}".encode()).hexdigest())
    chosen = set(keep)
    for record_id in ordered:
        if len(chosen) >= size:
            break
        chosen.add(record_id)
    return tuple(sorted(chosen))


# --- ranking ----------------------------------------------------------------


def _norm(values: array[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _cosine(left: array[float], left_norm: float, right: array[float], right_norm: float) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _average_ranks(scores: list[float]) -> list[float]:
    """Ranks with ties averaged, so a tie pulls neither way.

    Breaking ties by position here would make the correlation depend on the
    identifier order rather than on the vectors, which is the opposite of what
    the number is for.
    """
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    """Spearman correlation over two score vectors, or ``None`` when undefined.

    Undefined means one side has no variance at all -- every candidate tied --
    in which case there is no ordering to agree or disagree with, and inventing
    a 0 would read as disagreement rather than as absence.
    """
    if len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    count = len(left_ranks)
    left_mean = sum(left_ranks) / count
    right_mean = sum(right_ranks) / count
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_ranks, right_ranks, strict=True))
    left_spread = math.sqrt(sum((a - left_mean) ** 2 for a in left_ranks))
    right_spread = math.sqrt(sum((b - right_mean) ** 2 for b in right_ranks))
    if left_spread == 0.0 or right_spread == 0.0:
        return None
    return max(-1.0, min(1.0, covariance / (left_spread * right_spread)))


@dataclass(frozen=True)
class Ranking:
    """One query's view of one store."""

    ordered: tuple[str, ...]
    scores: list[float]
    boundary_tie: bool


def _rank(
    query: array[float],
    pool: tuple[str, ...],
    vectors: dict[str, array[float]],
    norms: dict[str, float],
    top_k: int,
) -> Ranking:
    query_norm = _norm(query)
    scores = [_cosine(query, query_norm, vectors[record_id], norms[record_id]) for record_id in pool]
    order = sorted(range(len(pool)), key=lambda index: (-scores[index], pool[index]))
    boundary = len(order) > top_k and scores[order[top_k - 1]] == scores[order[top_k]]
    return Ranking(ordered=tuple(pool[index] for index in order[:top_k]), scores=scores, boundary_tie=boundary)


def _reciprocal_rank(ordered: tuple[str, ...], expected: tuple[str, ...]) -> float:
    wanted = set(expected)
    for position, record_id in enumerate(ordered, start=1):
        if record_id in wanted:
            return 1.0 / position
    return 0.0


# --- the comparison ---------------------------------------------------------


def _build_checks(metrics: dict[str, float], thresholds: QualityThresholds, sample_size: int) -> list[Check]:
    checks: list[Check] = [
        Check(
            name="minimum_sample_size",
            passed=sample_size >= thresholds.minimum_sample_size,
            actual=float(sample_size),
            threshold=float(thresholds.minimum_sample_size),
            detail=f"{sample_size} queries were evaluated",
        )
    ]
    if thresholds.minimum_top_k_overlap is not None and "top_k_overlap" in metrics:
        actual = metrics["top_k_overlap"]
        checks.append(
            Check(
                name="minimum_top_k_overlap",
                passed=actual >= thresholds.minimum_top_k_overlap,
                actual=actual,
                threshold=float(thresholds.minimum_top_k_overlap),
                detail="mean share of the old top-k that is still in the new top-k",
            )
        )
    if thresholds.minimum_rank_correlation is not None and "rank_correlation" in metrics:
        actual = metrics["rank_correlation"]
        checks.append(
            Check(
                name="minimum_rank_correlation",
                passed=actual >= thresholds.minimum_rank_correlation,
                actual=actual,
                threshold=float(thresholds.minimum_rank_correlation),
                detail="mean Spearman correlation over the candidate pool",
            )
        )
    for name, metric, before, label in (
        ("maximum_hit_rate_drop", "hit_rate", "hit_rate_before", "hit rate"),
        ("maximum_mrr_drop", "mrr", "mrr_before", "MRR"),
    ):
        cap = getattr(thresholds, name)
        if cap is None or metric not in metrics or before not in metrics:
            continue
        drop = metrics[before] - metrics[metric]
        checks.append(
            Check(
                name=name,
                passed=drop <= cap,
                actual=drop,
                threshold=float(cap),
                detail=f"{label} moved from {metrics[before]:.4f} to {metrics[metric]:.4f}",
            )
        )
    return checks


def _metric_checks(checks: list[Check]) -> list[Check]:
    """The checks that judge the comparison.

    The sample-size check is a gate on whether to report at all, not a
    judgement of what was found, so a run carrying only that one has measured
    and judged nothing.
    """
    return [check for check in checks if check.name != "minimum_sample_size"]


def _missing_floors(query_set: QuerySet) -> str:
    """What an operator can declare to turn these metrics into a verdict.

    A result with no verdict has to say this, or it reads as a dead end rather
    than as a declaration the operator can still supply.
    """
    guidance = (
        "No agreement floor was declared, so there is no verdict. Declare minimum_top_k_overlap or minimum_rank_correlation to get one."
    )
    if query_set.mode == RECORD_ANCHORED:
        return f"{guidance} The drop caps need expected records, which anchors do not have."
    return f"{guidance} Expected records would also bring the drop caps into play."


def _text_embedders(before: VectorStore, after: VectorStore) -> tuple[Embedder, Embedder]:
    for store in (before, after):
        if store.model.id not in BUILT_IN_MODEL_IDS:
            raise QualityError(
                f"Text queries need both models re-run offline, and store {store.store_id!r} declares "
                f"{store.model.id!r}, which this package does not implement. Use record-anchored queries, "
                "whose query vectors are already in the stores."
            )
    try:
        return build_built_in_embedder(before.model), build_built_in_embedder(after.model)
    except EmbedderError as exc:
        raise QualityError(str(exc)) from None


def compare_stores(
    *,
    before: VectorStore,
    after: VectorStore,
    query_set: QuerySet,
    thresholds: QualityThresholds | None = None,
    candidate_pool: int = DEFAULT_CANDIDATE_POOL,
    seed: str = DEFAULT_SEED,
) -> QualityResult:
    """Rank both stores for every query and report what agreed.

    The two stores must hold the same records. A rebuilt store that dropped or
    invented records is not a different embedding of the same thing, and
    comparing neighborhoods across different populations would produce a number
    with no meaning rather than a low one.
    """
    limits = thresholds if thresholds is not None else QualityThresholds()
    if not 1 <= candidate_pool <= MAX_CANDIDATE_POOL:
        raise QualityError(f"candidate_pool must be between 1 and {MAX_CANDIDATE_POOL}, not {candidate_pool}.")
    shared = set(before.vectors) & set(after.vectors)
    if not shared:
        raise QualityError("The two stores share no record identifiers, so there is nothing to compare.")

    anchors = tuple(anchor for anchor in query_set.anchors if anchor in shared)
    skipped = query_set.declared_size - (len(anchors) if query_set.mode == RECORD_ANCHORED else len(query_set.queries))
    keep = set(anchors)
    pool = _draw_pool(tuple(sorted(shared)), size=candidate_pool, seed=query_set.seed or seed, keep=keep)

    queries = len(anchors) if query_set.mode == RECORD_ANCHORED else len(query_set.queries)
    operations = queries * len(pool) * before.model.dimensions * 2
    if operations > MAX_SIMILARITY_OPERATIONS:
        raise QualityError(
            f"This comparison would need about {operations} multiplications, over the "
            f"{MAX_SIMILARITY_OPERATIONS} budget. Lower the query count or the candidate pool."
        )

    basis = NEIGHBORHOOD_AGREEMENT
    if query_set.mode == TEXT and query_set.has_expected_records:
        basis = RETRIEVAL_RELEVANCE

    top_k = limits.top_k
    before_norms = {record_id: _norm(before.vectors[record_id]) for record_id in pool}
    after_norms = {record_id: _norm(after.vectors[record_id]) for record_id in pool}

    overlaps: list[float] = []
    correlations: list[float] = []
    hits_before: list[float] = []
    hits_after: list[float] = []
    reciprocal_before: list[float] = []
    reciprocal_after: list[float] = []
    boundary_ties = 0
    undefined = False
    evaluated = 0
    effective_k_capped = False

    plan: list[tuple[array[float], array[float], tuple[str, ...], tuple[str, ...]]] = []
    if query_set.mode == RECORD_ANCHORED:
        for anchor in anchors:
            # An anchor is its own nearest neighbour in both stores, so leaving
            # it in its own candidate list would add a guaranteed agreement to
            # every query and quietly raise every overlap.
            candidates = tuple(record_id for record_id in pool if record_id != anchor)
            plan.append((before.vectors[anchor], after.vectors[anchor], candidates, ()))
    else:
        # Built before any ranking, so a model this package cannot re-run
        # fails the whole comparison rather than half of it.
        before_embedder, after_embedder = _text_embedders(before, after)
        for text, expected in query_set.queries:
            plan.append(
                (
                    array("d", before_embedder.embed(text)),
                    array("d", after_embedder.embed(text)),
                    pool,
                    expected,
                )
            )

    for before_query, after_query, candidates, expected in plan:
        if len(candidates) < 2:
            skipped += 1
            continue
        effective_k = min(top_k, len(candidates))
        effective_k_capped = effective_k_capped or effective_k < top_k
        before_ranking = _rank(before_query, candidates, before.vectors, before_norms, effective_k)
        after_ranking = _rank(after_query, candidates, after.vectors, after_norms, effective_k)
        evaluated += 1
        boundary_ties += int(before_ranking.boundary_tie or after_ranking.boundary_tie)
        overlaps.append(len(set(before_ranking.ordered) & set(after_ranking.ordered)) / effective_k)
        correlation = _spearman(before_ranking.scores, after_ranking.scores)
        if correlation is None:
            undefined = True
        else:
            correlations.append(correlation)
        if expected:
            wanted = set(expected)
            hits_before.append(float(bool(wanted & set(before_ranking.ordered))))
            hits_after.append(float(bool(wanted & set(after_ranking.ordered))))
            reciprocal_before.append(_reciprocal_rank(before_ranking.ordered, expected))
            reciprocal_after.append(_reciprocal_rank(after_ranking.ordered, expected))

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics: dict[str, float] = {}
    if evaluated:
        metrics["top_k_overlap"] = mean(overlaps)
        if correlations:
            metrics["rank_correlation"] = mean(correlations)
        if hits_after:
            metrics["hit_rate"] = mean(hits_after)
            metrics["hit_rate_before"] = mean(hits_before)
            metrics["mrr"] = mean(reciprocal_after)
            metrics["mrr_before"] = mean(reciprocal_before)

    sufficient = evaluated >= limits.minimum_sample_size
    common: dict[str, Any] = {
        "mode": query_set.mode,
        "metric_basis": basis,
        "anchor_selection": query_set.selection,
        "anchor_seed": query_set.seed if query_set.selection == SEEDED_RANDOM else None,
        "query_set_size": evaluated,
        "top_k": top_k,
        "candidate_pool_size": len(pool),
        "boundary_ties": boundary_ties,
        "queries_skipped": max(skipped, 0),
        "correlation_undefined": undefined,
        "top_k_capped": effective_k_capped,
    }

    if evaluated == 0:
        return QualityResult(
            status=NOT_MEASURED,
            sufficient_evidence=False,
            metrics={},
            notes="No query could be evaluated, so nothing was measured. This is not a passing result.",
            **common,
        )
    if not sufficient:
        return QualityResult(
            status=NOT_MEASURED,
            sufficient_evidence=False,
            metrics={},
            notes=(
                f"{evaluated} queries is under the declared minimum of {limits.minimum_sample_size}, "
                "so the comparison was run but is not reported as a measurement."
            ),
            **common,
        )

    checks = _build_checks(metrics, limits, evaluated)
    judged = bool(_metric_checks(checks))
    if not judged:
        status = UNJUDGED
    elif all(check.passed for check in checks):
        status = SAMPLED
    else:
        status = BELOW_THRESHOLD
    note = (
        "Neighborhood agreement between the two stores. Not a measurement of search quality."
        if basis == NEIGHBORHOOD_AGREEMENT
        else "Retrieval against operator-declared expected records."
    )
    if not judged:
        note = f"{_missing_floors(query_set)} {note}"
    return QualityResult(
        status=status,
        sufficient_evidence=True,
        metrics=metrics,
        checks=tuple(checks),
        thresholds={check.name: check.threshold for check in checks},
        notes=note,
        **common,
    )


def attach_quality(record_path: Path, result: QualityResult) -> dict[str, Any]:
    """Write a comparison into an existing ``re-embed-record.v1`` and return it.

    The record is rewritten whole rather than patched in place, so a reader
    never sees a half-written document, and the quality block replaces any
    earlier one rather than merging with it: two comparisons are two answers,
    not one.
    """
    path = Path(record_path)
    try:
        document = load_json_document(read_bounded_text(path, MAX_QUERY_SET_BYTES), path.name)
    except EvidenceError as exc:
        raise QualityError(str(exc)) from None
    if document.get("schema_version") != "re-embed-record.v1":
        raise QualityError(f"Expected a re-embed-record.v1, found {document.get('schema_version')!r}.")
    document["quality"] = result.to_document()
    payload = json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return document
