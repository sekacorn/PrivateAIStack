"""A pre-run cost estimate for a re-embedding run: ``re-embed-estimate.v1``.

An operator deciding whether to rebuild a store wants to know what it will cost
before committing to it. AIMeter supplies the arithmetic but not a forecast:
``calculate_local_inference_cost`` requires an *observed*
``performance.latency_ms`` greater than zero, and ``calculate_provider_cost`` is
token-driven. Neither can be called before a run has happened.

So the honest construction is a **calibration sample**. A small, configurable
number of retained records is embedded for real, its latency and token counts
are measured, and the measurement is extrapolated to the full retained count.
The exact part and the estimated part are reported separately:

* ``records_to_embed`` is exact. It comes from the evidence document.
* tokens, latency and any cost are extrapolated from the sample, and every
  figure resting on the sample says so.

A sample of zero produces **no estimate at all** rather than a fabricated one.

The estimate is a forecast, not evidence. It is never read by ``exit-report``,
it carries no digest of itself, and it is deliberately not reproducible: wall
clock latency depends on the machine and its load at the time. Only the counts
in it are stable.

Cost in money needs AIMeter, which this project does not depend on. When an
operator supplies a local cost profile and ``aimeter-oss`` is importable, its
``calculate_local_inference_cost`` is used and credited. Otherwise the estimate
reports records, tokens and latency and says plainly why there is no money
figure. Nothing here reimplements AIMeter's arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from private_ai_stack.reembed.checkpoint import write_json_atomically
from private_ai_stack.reembed.embedders import Embedder, ModelIdentity, build_built_in_embedder
from private_ai_stack.reembed.evidence import SourceRecord, SourceStore, load_source_store

SCHEMA_VERSION: Final = "re-embed-estimate.v1"
ESTIMATE_FILENAME: Final = "re-embed-estimate.json"
DEFAULT_SAMPLE_SIZE: Final = 16
MAX_SAMPLE_SIZE: Final = 1000
DEFAULT_SAMPLE_SEED: Final = "re-embed-estimate"
SAMPLE_SELECTION: Final = "sha256(seed + colon + record_id) ascending"

NOT_ESTIMATED: Final = "not_estimated"
ESTIMATED: Final = "estimated"
WHITESPACE_TOKENIZER: Final = "whitespace (the embedder does not expose its tokenisation)"


class EstimateError(ValueError):
    """Raised when an estimate cannot be produced as asked."""


@dataclass(frozen=True)
class Calibration:
    """What embedding a sample actually measured."""

    sample_size: int
    tokens: int
    source_bytes: int
    latency_ms: float
    tokenizer: str
    sample_seed: str

    @property
    def tokens_per_record(self) -> float:
        return self.tokens / self.sample_size

    @property
    def latency_ms_per_record(self) -> float:
        return self.latency_ms / self.sample_size

    def to_document(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "sample_selection": SAMPLE_SELECTION,
            "sample_seed": self.sample_seed,
            "tokenizer": self.tokenizer,
            "measured_tokens": self.tokens,
            "measured_source_bytes": self.source_bytes,
            "measured_latency_ms": round(self.latency_ms, 6),
            "tokens_per_record": round(self.tokens_per_record, 6),
            "latency_ms_per_record": round(self.latency_ms_per_record, 6),
        }


@dataclass(frozen=True)
class ReEmbedEstimate:
    """A forecast for a run that has not happened."""

    source_store_id: str
    source_evidence_sha256: str
    target_model: ModelIdentity
    records_to_embed: int
    missing_skipped: int
    deleted_by_policy_skipped: int
    calibration: Calibration | None
    cost: dict[str, Any]
    limits: tuple[str, ...]

    @property
    def estimated_tokens(self) -> int | None:
        if self.calibration is None:
            return None
        return round(self.calibration.tokens_per_record * self.records_to_embed)

    @property
    def estimated_latency_ms(self) -> float | None:
        if self.calibration is None:
            return None
        return self.calibration.latency_ms_per_record * self.records_to_embed

    def to_document(self) -> dict[str, Any]:
        latency = self.estimated_latency_ms
        projection = (
            None
            if self.calibration is None
            else {
                "status": ESTIMATED,
                "estimated_tokens": self.estimated_tokens,
                "estimated_latency_ms": None if latency is None else round(latency, 6),
            }
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "source_store": {
                "store_id": self.source_store_id,
                "store_evidence_sha256": self.source_evidence_sha256,
            },
            "target_model": {
                "id": self.target_model.id,
                "revision": self.target_model.revision,
                "dimensions": self.target_model.dimensions,
            },
            "records_to_embed": self.records_to_embed,
            "records_skipped": {
                "missing": self.missing_skipped,
                "deleted_by_policy": self.deleted_by_policy_skipped,
            },
            "calibration": None if self.calibration is None else self.calibration.to_document(),
            "projection": projection,
            "cost": self.cost,
            "egress": {
                "remote_provider_used": False,
                "records_transmitted": 0,
                "raw_source_text_transmitted": False,
            },
            "limits": list(self.limits),
        }


def _count_tokens(embedder: Embedder, text: str) -> tuple[int, str]:
    """Count the tokens the embedder itself will hash, and say how.

    Asking the embedder keeps the count from drifting from the vector it
    describes. An embedder that does not expose its tokenisation gets a
    whitespace count, labelled as such so nobody mistakes it for the model's
    own.
    """
    method = getattr(embedder, "tokens", None)
    if callable(method):
        produced = method(text)
        if isinstance(produced, list) and all(isinstance(item, str) for item in produced):
            return len(produced), f"{embedder.identity.id}@{embedder.identity.revision}"
    return len(text.split()), WHITESPACE_TOKENIZER


def _sample(store: SourceStore, sample_size: int, seed: str) -> list[SourceRecord]:
    """Pick records deterministically but without following record_id order.

    Taking the first N by identifier would be reproducible and biased: nothing
    says identifiers are unrelated to record size. Ordering by a seeded digest
    is reproducible and spreads the sample, and the seed is recorded so a
    reader can repeat the selection.
    """
    ordered = sorted(
        store.retained,
        key=lambda record: hashlib.sha256(f"{seed}:{record.record_id}".encode()).hexdigest(),
    )
    return ordered[:sample_size]


def _local_cost(
    calibration: Calibration | None,
    records_to_embed: int,
    cost_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Delegate the money arithmetic to AIMeter, or say why there is none."""
    if cost_profile is None:
        return {"status": NOT_ESTIMATED, "reason": "No local cost profile was supplied."}
    if calibration is None or records_to_embed == 0:
        return {
            "status": NOT_ESTIMATED,
            "reason": "No calibration sample was embedded, so there is nothing to extrapolate.",
        }
    try:
        from ai_meter import (  # noqa: PLC0415
            InfrastructureProfile,
            UsageRecord,
            __version__,
            calculate_local_inference_cost,
        )
    except ImportError:
        return {
            "status": NOT_ESTIMATED,
            "reason": (
                "A local cost profile was supplied but aimeter-oss is not installed, and this project will not reimplement its arithmetic."
            ),
        }
    profile_data = cost_profile.get("profile", cost_profile)
    currency = profile_data.get("currency") if isinstance(profile_data, dict) else None
    if not isinstance(currency, str) or not currency:
        raise EstimateError("A local cost profile must declare a currency.")
    latency_ms = calibration.latency_ms_per_record * records_to_embed
    if latency_ms <= 0:
        return {
            "status": NOT_ESTIMATED,
            "reason": "The calibration sample measured no elapsed time, and AIMeter requires a positive duration.",
        }
    record = UsageRecord(
        {
            "record_id": "re-embed-estimate",
            "cost": {"currency": currency},
            "performance": {"latency_ms": repr(latency_ms)},
        }
    )
    result = dict(calculate_local_inference_cost(record, InfrastructureProfile(cost_profile)))
    result["total_cost"] = str(result.get("total_cost"))
    result["components"] = {key: str(value) for key, value in dict(result.get("components", {})).items()}
    result["source"] = f"aimeter-oss {__version__} calculate_local_inference_cost"
    result["extrapolated_latency_ms"] = round(latency_ms, 6)
    _annotate_precision(result, profile_data if isinstance(profile_data, dict) else {}, currency)
    return result


def _positive_rates(profile_data: dict[str, Any]) -> bool:
    """Whether any assumption in the profile could produce a cost above zero.

    This is what separates the two kinds of zero. A profile whose every rate is
    zero says the compute is free, and its total really is nothing. A profile
    with a positive rate and a total of zero means the duration was too short
    for the smallest unit AIMeter prints.
    """
    groups = (
        ("compute", ("gpu_hourly_cost", "cpu_hourly_cost", "memory_hourly_cost")),
        ("energy", ("average_watts", "electricity_per_kwh")),
    )
    for group, keys in groups:
        section = profile_data.get(group)
        if not isinstance(section, dict):
            continue
        for key in keys:
            try:
                if Decimal(str(section.get(key, "0"))) > 0:
                    return True
            except (ArithmeticError, ValueError):
                continue
    return False


def _annotate_precision(result: dict[str, Any], profile_data: dict[str, Any], currency: str) -> None:
    """Say "below reporting precision" rather than printing a misleading zero.

    Embedding a handful of short records with a hashing scheme takes
    microseconds, so AIMeter's total rounds to ``0.000000`` and a reader could
    take that for free. It is not free, it is smaller than the smallest unit
    reported, and those are different claims. The bound comes from the number
    of decimals AIMeter itself printed, so it stays correct if that precision
    ever changes.

    A genuine zero stays a zero: a profile with no positive rate at all says
    the compute costs nothing, and there is nothing imprecise about that.
    """
    printed = str(result.get("total_cost", ""))
    try:
        if Decimal(printed) != 0:
            return
    except (ArithmeticError, ValueError):
        return
    if not _positive_rates(profile_data):
        result["total_cost_precision"] = "exact_zero"
        result["total_cost_detail"] = f"The profile declares no positive rate, so the cost really is 0 {currency}."
        return
    decimals = len(printed.partition(".")[2])
    smallest = Decimal(1).scaleb(-decimals) if decimals else Decimal(1)
    result["total_cost_precision"] = "below_reporting_precision"
    result["total_cost_bound"] = f"< {smallest:f} {currency}"
    result["total_cost_detail"] = (
        f"The extrapolated cost is above zero but below {smallest:f} {currency}, the smallest unit AIMeter reports. It is not free."
    )


def _limits(calibration: Calibration | None, records_to_embed: int, cost: dict[str, Any]) -> tuple[str, ...]:
    limits = ["records_to_embed is exact; every other figure is extrapolated and is not a quotation."]
    if calibration is None:
        limits.append("No calibration sample was embedded, so no tokens, latency or cost are reported.")
        return tuple(limits)
    limits.append(f"Tokens and latency come from a calibration sample of {calibration.sample_size} of {records_to_embed} records.")
    limits.append("Measured latency reflects this machine under its load at the time, and is not portable to another.")
    if cost.get("total_cost_precision") == "below_reporting_precision":
        limits.append(
            f"The cost is above zero but {cost.get('total_cost_bound')}, the smallest unit AIMeter reports. "
            "It prints as zero and is not free."
        )
    if cost.get("status") == ESTIMATED:
        limits.append(
            "Cost rests on the supplied profile's assumptions. AIMeter's own templates state that they are "
            "examples to be replaced with organisation-specific figures."
        )
    return tuple(limits)


def estimate_re_embed(
    *,
    source_document: Path,
    source_store_id: str,
    target_model: ModelIdentity,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sample_seed: str = DEFAULT_SAMPLE_SEED,
    cost_profile: dict[str, Any] | None = None,
    embedder: Embedder | None = None,
    output: Path | None = None,
) -> ReEmbedEstimate:
    """Measure a calibration sample and extrapolate it to the whole store.

    Nothing is written to a rebuilt store and no vector is kept: the sample is
    embedded in memory, one record at a time, and discarded, so the memory
    bound is one record rather than one batch. ``output`` optionally writes the
    estimate document, which is a forecast rather than evidence.
    """
    if not 0 <= sample_size <= MAX_SAMPLE_SIZE:
        raise EstimateError(f"Calibration sample size must be between 0 and {MAX_SAMPLE_SIZE}.")
    if not sample_seed:
        raise EstimateError("A calibration sample seed is required, so the selection can be repeated.")
    store = load_source_store(source_document, store_id=source_store_id)
    worker = embedder if embedder is not None else build_built_in_embedder(target_model)
    if worker.identity.triple != target_model.triple:
        raise EstimateError(f"Embedder identity {worker.identity.triple} does not match the declared {target_model.triple}.")

    records_to_embed = len(store.retained)
    effective = min(sample_size, records_to_embed)
    calibration: Calibration | None = None
    if effective > 0:
        tokens = 0
        source_bytes = 0
        tokenizer = WHITESPACE_TOKENIZER
        elapsed_ns = 0
        for record in _sample(store, effective, sample_seed):
            text = store.read_text(record)
            source_bytes += len(text.encode("utf-8"))
            started = time.perf_counter_ns()
            vector = worker.embed(text)
            elapsed_ns += time.perf_counter_ns() - started
            if len(vector) != target_model.dimensions:
                raise EstimateError(
                    f"Embedder produced {len(vector)} dimensions for {record.record_id!r}, "
                    f"but the declared model is {target_model.dimensions}."
                )
            counted, tokenizer = _count_tokens(worker, text)
            tokens += counted
        calibration = Calibration(
            sample_size=effective,
            tokens=tokens,
            source_bytes=source_bytes,
            latency_ms=elapsed_ns / 1_000_000,
            tokenizer=tokenizer,
            sample_seed=sample_seed,
        )

    cost = _local_cost(calibration, records_to_embed, cost_profile)
    estimate = ReEmbedEstimate(
        source_store_id=store.store_id,
        source_evidence_sha256=store.document_sha256,
        target_model=target_model,
        records_to_embed=records_to_embed,
        missing_skipped=store.count("missing"),
        deleted_by_policy_skipped=store.count("deleted_by_policy"),
        calibration=calibration,
        cost=cost,
        limits=_limits(calibration, records_to_embed, cost),
    )
    if output is not None:
        write_json_atomically(output, estimate.to_document())
    return estimate


def load_cost_profile(path: Path) -> dict[str, Any]:
    """Read an operator-supplied local cost profile from JSON.

    AIMeter reads these from YAML, which would be a new dependency here, so
    this takes the same structure as JSON. ``local_cost_profile_catalog()``
    emits exactly this shape.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EstimateError(f"Cannot read cost profile {path.name}: {exc}") from None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EstimateError(f"Cost profile {path.name} is not valid JSON: {exc}") from None
    if not isinstance(document, dict):
        raise EstimateError(f"Cost profile {path.name} must be a JSON object.")
    return document
