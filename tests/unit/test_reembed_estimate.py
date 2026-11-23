"""The pre-run estimate: what it measures, what it refuses to claim."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from private_ai_stack.reembed.embedders import ModelIdentity
from private_ai_stack.reembed.estimate import (
    DEFAULT_SAMPLE_SEED,
    MAX_SAMPLE_SIZE,
    SCHEMA_VERSION,
    WHITESPACE_TOKENIZER,
    EstimateError,
    estimate_re_embed,
    load_cost_profile,
)
from private_ai_stack.reembed.evidence import EvidenceError
from tests.unit.reembed_support import build_source_store

MODEL = ModelIdentity(id="privateaistack.hashed-bow.sha256-signed", revision="1", dimensions=16)
SCHEMA = Path(__file__).resolve().parents[2] / "private_ai_stack" / "reembed" / "schemas" / "re-embed-estimate.schema.json"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "reembed" / "re-embed-estimate.json"

TEXTS = {
    "procedure-escalation": "Escalate to the on-call engineer when the queue depth exceeds 40.",
    "procedure-password-reset": "Reset the password through the self-service portal.",
    "procedure-refund": "Refunds over 500 need a supervisor approval and an audit entry.",
    "procedure-sla": "Acknowledge within one hour and resolve within one business day.",
    "procedure-handover": "Summarise the open threads before handing the queue over.",
}

PROFILE: dict[str, Any] = {
    "profile": {
        "id": "example-workstation",
        "currency": "USD",
        "compute": {"gpu_hourly_cost": "0.18", "cpu_hourly_cost": "0.04", "memory_hourly_cost": "0.01"},
        "energy": {"average_watts": "280", "electricity_per_kwh": "0.16"},
        "allocation": {"overhead_percentage": "10"},
    }
}


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return build_source_store(
        tmp_path / "evidence",
        texts=TEXTS,
        missing=("procedure-gone",),
        deleted_by_policy=("procedure-purged",),
        dimensions=8,
    )


def _estimate(source_document: Path, **overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "source_document": source_document,
        "source_store_id": "support-memory",
        "target_model": MODEL,
        "sample_size": 3,
    }
    arguments.update(overrides)
    return estimate_re_embed(**arguments)


def test_the_exact_count_comes_from_the_document_not_the_sample(source: Path) -> None:
    estimate = _estimate(source)
    assert estimate.records_to_embed == len(TEXTS)
    assert estimate.calibration is not None
    assert estimate.calibration.sample_size == 3


def test_records_with_no_text_are_counted_as_skipped_not_embedded(source: Path) -> None:
    estimate = _estimate(source)
    assert estimate.missing_skipped == 1
    assert estimate.deleted_by_policy_skipped == 1
    assert estimate.records_to_embed == 5


def test_tokens_are_counted_by_the_model_that_will_embed_them(source: Path) -> None:
    estimate = _estimate(source, sample_size=len(TEXTS))
    assert estimate.calibration is not None
    assert estimate.calibration.tokenizer == f"{MODEL.id}@{MODEL.revision}"
    expected = sum(len(text.lower().split()) for text in TEXTS.values())
    assert estimate.calibration.tokens == expected


def test_a_full_sample_projects_the_measured_total_exactly(source: Path) -> None:
    """With every record sampled there is nothing to extrapolate, so the
    projection must equal the measurement rather than drift through a mean."""
    estimate = _estimate(source, sample_size=len(TEXTS))
    assert estimate.calibration is not None
    assert estimate.estimated_tokens == estimate.calibration.tokens


def test_the_projection_scales_the_per_record_mean(source: Path) -> None:
    estimate = _estimate(source, sample_size=2)
    assert estimate.calibration is not None
    assert estimate.estimated_tokens == round(estimate.calibration.tokens / 2 * 5)


def test_a_sample_of_zero_reports_no_estimate_rather_than_a_guess(source: Path) -> None:
    estimate = _estimate(source, sample_size=0)
    assert estimate.calibration is None
    assert estimate.estimated_tokens is None
    assert estimate.estimated_latency_ms is None
    document = estimate.to_document()
    assert document["calibration"] is None
    assert document["projection"] is None
    assert any("no calibration sample" in limit.lower() for limit in document["limits"])


def test_a_sample_larger_than_the_store_is_capped_not_refused(source: Path) -> None:
    estimate = _estimate(source, sample_size=500)
    assert estimate.calibration is not None
    assert estimate.calibration.sample_size == len(TEXTS)


def test_the_sample_selection_is_reproducible_under_the_same_seed(source: Path) -> None:
    first = _estimate(source, sample_size=2)
    second = _estimate(source, sample_size=2)
    assert first.calibration is not None and second.calibration is not None
    assert first.calibration.tokens == second.calibration.tokens
    assert first.calibration.source_bytes == second.calibration.source_bytes


def test_a_different_seed_can_select_a_different_sample(source: Path) -> None:
    """Not that it must differ for any one store, but that the seed is used."""
    seeds = {_estimate(source, sample_size=1, sample_seed=seed).calibration.source_bytes for seed in ("a", "b", "c", "d")}  # type: ignore[union-attr]
    assert len(seeds) > 1


def test_the_sample_is_not_simply_the_first_records_by_identifier(source: Path) -> None:
    estimate = _estimate(source, sample_size=1)
    assert estimate.calibration is not None
    first_by_id = sorted(TEXTS)[0]
    assert estimate.calibration.source_bytes != len(TEXTS[first_by_id].encode("utf-8"))


def test_the_seed_is_recorded_so_the_selection_can_be_repeated(source: Path) -> None:
    document = _estimate(source, sample_size=2).to_document()
    assert document["calibration"]["sample_seed"] == DEFAULT_SAMPLE_SEED
    assert "record_id" in document["calibration"]["sample_selection"]


def test_no_store_is_written_and_no_vector_is_kept(source: Path, tmp_path: Path) -> None:
    before = {path for path in tmp_path.rglob("*")}
    _estimate(source, sample_size=len(TEXTS))
    assert {path for path in tmp_path.rglob("*")} == before


def test_the_estimate_can_be_written_as_a_document(source: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / "re-embed-estimate.json"
    estimate = _estimate(source, sample_size=2, output=output)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == estimate.to_document()
    assert written["schema_version"] == SCHEMA_VERSION


def test_the_document_never_claims_cost_without_a_profile(source: Path) -> None:
    document = _estimate(source, sample_size=2).to_document()
    assert document["cost"]["status"] == "not_estimated"
    assert "no local cost profile" in document["cost"]["reason"].lower()


def test_the_document_never_claims_egress(source: Path) -> None:
    document = _estimate(source, sample_size=2).to_document()
    assert document["egress"] == {
        "remote_provider_used": False,
        "records_transmitted": 0,
        "raw_source_text_transmitted": False,
    }


def test_the_limits_always_say_the_figure_is_not_a_quotation(source: Path) -> None:
    for size in (0, 1, len(TEXTS)):
        limits = _estimate(source, sample_size=size).to_document()["limits"]
        assert any("not a quotation" in limit for limit in limits)
        assert any("exact" in limit for limit in limits)


def test_the_limits_name_the_sample_size_and_the_population(source: Path) -> None:
    limits = _estimate(source, sample_size=2).to_document()["limits"]
    assert any("2 of 5 records" in limit for limit in limits)


def test_the_limits_warn_that_latency_is_not_portable(source: Path) -> None:
    limits = _estimate(source, sample_size=2).to_document()["limits"]
    assert any("not portable to another" in limit for limit in limits)


def test_a_cost_profile_without_aimeter_says_so_instead_of_improvising(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ai_meter", None)
    document = _estimate(source, sample_size=2, cost_profile=PROFILE).to_document()
    assert document["cost"]["status"] == "not_estimated"
    assert "aimeter-oss is not installed" in document["cost"]["reason"]
    assert "reimplement" in document["cost"]["reason"]


def test_cost_is_delegated_to_aimeter_and_credited(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in for aimeter-oss, which this project does not depend on.

    The point under test is the delegation and the attribution, not AIMeter's
    arithmetic: the duration handed over must be the extrapolated one, and the
    result must say which package produced the number.
    """
    seen: dict[str, Any] = {}

    class _Record:
        def __init__(self, data: dict[str, Any]) -> None:
            self.data = data

    class _Profile:
        def __init__(self, data: dict[str, Any]) -> None:
            self.data = data

    def _calculate(record: Any, profile: Any) -> dict[str, Any]:
        seen["latency_ms"] = record.data["performance"]["latency_ms"]
        seen["profile_id"] = profile.data["profile"]["id"]
        return {
            "status": "estimated",
            "currency": record.data["cost"]["currency"],
            "total_cost": "0.000123",
            "components": {"gpu": "0.0001"},
            "assumptions": profile.data["profile"]["id"],
        }

    module = types.ModuleType("ai_meter")
    module.UsageRecord = _Record  # type: ignore[attr-defined]
    module.InfrastructureProfile = _Profile  # type: ignore[attr-defined]
    module.calculate_local_inference_cost = _calculate  # type: ignore[attr-defined]
    module.__version__ = "9.9.9"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_meter", module)

    estimate = _estimate(source, sample_size=2, cost_profile=PROFILE)
    document = estimate.to_document()
    assert document["cost"]["status"] == "estimated"
    assert document["cost"]["total_cost"] == "0.000123"
    assert document["cost"]["source"] == "aimeter-oss 9.9.9 calculate_local_inference_cost"
    assert seen["profile_id"] == "example-workstation"
    assert float(seen["latency_ms"]) == pytest.approx(estimate.estimated_latency_ms)
    assert any("AIMeter's own templates" in limit for limit in document["limits"])


def test_a_cost_profile_with_no_currency_is_refused(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("ai_meter")
    module.UsageRecord = object  # type: ignore[attr-defined]
    module.InfrastructureProfile = object  # type: ignore[attr-defined]
    module.calculate_local_inference_cost = lambda record, profile: {}  # type: ignore[attr-defined]
    module.__version__ = "0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ai_meter", module)
    with pytest.raises(EstimateError, match="currency"):
        _estimate(source, sample_size=2, cost_profile={"profile": {"id": "x"}})


def test_a_cost_profile_with_no_sample_is_not_extrapolated(source: Path) -> None:
    document = _estimate(source, sample_size=0, cost_profile=PROFILE).to_document()
    assert document["cost"]["status"] == "not_estimated"
    assert "nothing to extrapolate" in document["cost"]["reason"]


@pytest.mark.parametrize("size", [-1, MAX_SAMPLE_SIZE + 1])
def test_an_out_of_range_sample_size_is_refused(source: Path, size: int) -> None:
    with pytest.raises(EstimateError, match="sample size"):
        _estimate(source, sample_size=size)


def test_an_empty_seed_is_refused(source: Path) -> None:
    with pytest.raises(EstimateError, match="seed"):
        _estimate(source, sample_seed="")


def test_an_unknown_target_model_is_refused(source: Path) -> None:
    from private_ai_stack.reembed.embedders import EmbedderError

    with pytest.raises(EmbedderError, match="not a built-in embedder"):
        _estimate(source, target_model=ModelIdentity(id="nope", revision="1", dimensions=16))


def test_an_absent_store_is_refused(source: Path) -> None:
    with pytest.raises(EvidenceError):
        _estimate(source, source_store_id="not-this-one")


def test_a_mislabelled_embedder_is_refused(source: Path) -> None:
    class _Wrong:
        @property
        def identity(self) -> ModelIdentity:
            return ModelIdentity(id="agentforge.hashed-bow.blake2b", revision="1", dimensions=16)

        def embed(self, text: str) -> list[float]:
            return [0.0] * 16

    with pytest.raises(EstimateError, match="does not match the declared"):
        _estimate(source, embedder=_Wrong())


def test_a_wrong_width_embedder_is_refused(source: Path) -> None:
    class _Narrow:
        @property
        def identity(self) -> ModelIdentity:
            return MODEL

        def embed(self, text: str) -> list[float]:
            return [0.0] * 4

    with pytest.raises(EstimateError, match="dimensions"):
        _estimate(source, embedder=_Narrow())


def test_an_embedder_without_a_tokenizer_gets_a_labelled_whitespace_count(source: Path) -> None:
    class _Opaque:
        @property
        def identity(self) -> ModelIdentity:
            return MODEL

        def embed(self, text: str) -> list[float]:
            return [0.0] * MODEL.dimensions

    estimate = _estimate(source, sample_size=len(TEXTS), embedder=_Opaque())
    assert estimate.calibration is not None
    assert estimate.calibration.tokenizer == WHITESPACE_TOKENIZER
    assert "whitespace" in estimate.calibration.tokenizer


def test_an_estimate_opens_no_socket(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def _forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("An estimate must not open a socket.")

    monkeypatch.setattr(socket, "socket", _forbidden)
    monkeypatch.setattr(socket, "create_connection", _forbidden)
    assert _estimate(source, sample_size=2).records_to_embed == 5


def test_a_cost_profile_is_read_from_json(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE), encoding="utf-8")
    assert load_cost_profile(path) == PROFILE


@pytest.mark.parametrize(
    ("content", "match"),
    [("[1, 2]", "must be a JSON object"), ("{", "not valid JSON")],
)
def test_a_malformed_cost_profile_is_refused(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "profile.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(EstimateError, match=match):
        load_cost_profile(path)


def test_an_absent_cost_profile_is_refused(tmp_path: Path) -> None:
    with pytest.raises(EstimateError, match="Cannot read cost profile"):
        load_cost_profile(tmp_path / "absent.json")


@pytest.mark.parametrize("name", ["re-embed-estimate.json", "re-embed-estimate-no-sample.json"])
def test_the_shipped_fixtures_match_the_schema_contract(name: str) -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    document = json.loads((FIXTURE.parent / name).read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION
    assert document["schema_version"] == SCHEMA_VERSION
    assert set(schema["required"]) <= set(document)
    assert set(document) <= set(schema["properties"])
    assert document["cost"]["status"] == "not_estimated"
    assert document["cost"]["reason"]
    assert document["limits"]


def test_the_two_fixtures_cover_the_sampled_and_unsampled_shapes() -> None:
    sampled = json.loads((FIXTURE.parent / "re-embed-estimate.json").read_text(encoding="utf-8"))
    unsampled = json.loads((FIXTURE.parent / "re-embed-estimate-no-sample.json").read_text(encoding="utf-8"))
    assert sampled["calibration"] is not None and sampled["projection"] is not None
    assert unsampled["calibration"] is None and unsampled["projection"] is None
    assert sampled["records_to_embed"] == unsampled["records_to_embed"]


def _aimeter(total: str) -> types.ModuleType:
    """A stand-in for aimeter-oss that returns a chosen total, as AIMeter prints it."""

    class _Record:
        def __init__(self, data: dict[str, Any]) -> None:
            self.data = data

    class _Profile:
        def __init__(self, data: dict[str, Any]) -> None:
            self.data = data

    def _calculate(record: Any, profile: Any) -> dict[str, Any]:
        return {
            "status": "estimated",
            "currency": record.data["cost"]["currency"],
            "total_cost": total,
            "components": {"gpu": total},
            "assumptions": "stand-in",
        }

    module = types.ModuleType("ai_meter")
    module.UsageRecord = _Record  # type: ignore[attr-defined]
    module.InfrastructureProfile = _Profile  # type: ignore[attr-defined]
    module.calculate_local_inference_cost = _calculate  # type: ignore[attr-defined]
    module.__version__ = "9.9.9"  # type: ignore[attr-defined]
    return module


FREE_PROFILE: dict[str, Any] = {
    "profile": {
        "id": "free-metal",
        "currency": "USD",
        "compute": {"gpu_hourly_cost": "0", "cpu_hourly_cost": "0", "memory_hourly_cost": "0"},
        "allocation": {"overhead_percentage": "0"},
    }
}


def test_a_cost_that_rounds_to_zero_is_reported_as_below_precision(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Microseconds of hashing really do cost something; printing 0.000000 would
    read as free, which is a different claim."""
    monkeypatch.setitem(sys.modules, "ai_meter", _aimeter("0.000000"))
    cost = _estimate(source, sample_size=2, cost_profile=PROFILE).to_document()["cost"]
    assert cost["status"] == "estimated"
    assert cost["total_cost_precision"] == "below_reporting_precision"
    assert cost["total_cost_bound"] == "< 0.000001 USD"
    assert "not free" in cost["total_cost_detail"]


def test_the_bound_follows_aimeters_own_precision(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bound is derived from the decimals AIMeter printed, not hardcoded."""
    monkeypatch.setitem(sys.modules, "ai_meter", _aimeter("0.00"))
    cost = _estimate(source, sample_size=2, cost_profile=PROFILE).to_document()["cost"]
    assert cost["total_cost_bound"] == "< 0.01 USD"


def test_the_limits_say_the_cost_is_not_free(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ai_meter", _aimeter("0.000000"))
    limits = _estimate(source, sample_size=2, cost_profile=PROFILE).to_document()["limits"]
    assert any("is not free" in limit for limit in limits)
    assert any("< 0.000001 USD" in limit for limit in limits)


def test_a_profile_with_no_positive_rate_keeps_a_true_zero(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A profile that says the compute is free is not imprecise, it is free."""
    monkeypatch.setitem(sys.modules, "ai_meter", _aimeter("0.000000"))
    document = _estimate(source, sample_size=2, cost_profile=FREE_PROFILE).to_document()
    cost = document["cost"]
    assert cost["total_cost"] == "0.000000"
    assert cost["total_cost_precision"] == "exact_zero"
    assert "total_cost_bound" not in cost
    assert not any("not free" in limit for limit in document["limits"])


def test_a_positive_cost_gets_no_precision_note(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ai_meter", _aimeter("1.250000"))
    cost = _estimate(source, sample_size=2, cost_profile=PROFILE).to_document()["cost"]
    assert cost["total_cost"] == "1.250000"
    assert "total_cost_precision" not in cost
    assert "total_cost_bound" not in cost


def test_an_energy_only_profile_still_counts_as_a_positive_rate(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ai_meter", _aimeter("0.000000"))
    profile = {
        "profile": {
            "id": "energy-only",
            "currency": "USD",
            "compute": {"gpu_hourly_cost": "0", "cpu_hourly_cost": "0", "memory_hourly_cost": "0"},
            "energy": {"average_watts": "280", "electricity_per_kwh": "0.16"},
        }
    }
    cost = _estimate(source, sample_size=2, cost_profile=profile).to_document()["cost"]
    assert cost["total_cost_precision"] == "below_reporting_precision"


def test_no_records_is_not_reported_as_below_precision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to embed is nothing to extrapolate, not a cost too small to print."""
    monkeypatch.setitem(sys.modules, "ai_meter", _aimeter("0.000000"))
    empty = build_source_store(tmp_path / "empty", texts={}, missing=("procedure-gone",), dimensions=8)
    document = estimate_re_embed(
        source_document=empty,
        source_store_id="support-memory",
        target_model=MODEL,
        sample_size=4,
        cost_profile=PROFILE,
    ).to_document()
    assert document["records_to_embed"] == 0
    assert document["cost"]["status"] == "not_estimated"
    assert "nothing to extrapolate" in document["cost"]["reason"]
    assert "total_cost_precision" not in document["cost"]


def test_an_unparsable_total_is_left_alone(source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If AIMeter ever returns something this code cannot read, it must not guess."""
    monkeypatch.setitem(sys.modules, "ai_meter", _aimeter("not-a-number"))
    cost = _estimate(source, sample_size=2, cost_profile=PROFILE).to_document()["cost"]
    assert cost["total_cost"] == "not-a-number"
    assert "total_cost_precision" not in cost


def test_the_cost_extra_declares_aimeter_in_the_suites_range() -> None:
    """The extra exists so `pip install privateaistack[cost]` is a documented
    request, and its range matches the suite's so one AIMeter resolves."""
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    assert extras["cost"] == ["aimeter-oss>=0.2.0b1,<0.3.0"]
    assert "aimeter-oss>=0.2.0b1,<0.3.0" in extras["all"]


def test_the_new_cost_fields_are_declared_in_the_schema() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    cost = schema["properties"]["cost"]["properties"]
    assert set(cost["total_cost_precision"]["enum"]) == {"below_reporting_precision", "exact_zero"}
    assert "total_cost_bound" in cost
    assert any(
        rule.get("if", {}).get("properties", {}).get("total_cost_precision", {}).get("const") == "below_reporting_precision"
        and rule["then"]["required"] == ["total_cost_bound"]
        for rule in schema["properties"]["cost"]["allOf"]
    )
