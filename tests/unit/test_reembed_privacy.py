"""The egress gate: a hosted run is impossible without a flag, and impossible to hide.

Negative test 6 of the phase 3b plan lives here. Every test is offline, and
several assert that by proving no socket could have been opened rather than by
trusting that none was.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from private_ai_stack.reembed import (
    DEFAULT_CLASSIFICATION,
    HOSTED,
    LOCAL,
    EgressError,
    ModelIdentity,
    ReEmbedError,
    provider_data_preflight,
    re_embed_store,
)
from private_ai_stack.reembed.embedders import build_built_in_embedder
from private_ai_stack.reembed.privacy import (
    BLOCKED_WARNING,
    TELEMETRY_WARNING,
    UNREDACTED_WARNING,
    local_egress,
)
from tests.unit.reembed_support import build_source_store

MODEL = ModelIdentity(id="privateaistack.hashed-bow.sha256-signed", revision="1", dimensions=8)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return build_source_store(tmp_path / "evidence", dimensions=8)


def _run(source: Path, target: Path, **kwargs: Any) -> Any:
    return re_embed_store(
        source_document=source,
        source_store_id="support-memory",
        target_directory=target,
        target_store_id="support-memory-rebuilt",
        target_model=MODEL,
        availability="open_weight_self_hostable",
        **kwargs,
    )


class NoSocket:
    """A socket type that fails the test if anything tries to use it."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("A re-embedding run must not open a socket.")


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make opening a socket an immediate test failure, not a silent success."""
    monkeypatch.setattr(socket, "socket", NoSocket)
    monkeypatch.setattr(socket, "create_connection", NoSocket)
    monkeypatch.setattr(socket, "getaddrinfo", NoSocket)


# --- the default ------------------------------------------------------------


def test_the_default_sends_nothing(no_network: None) -> None:
    gate = local_egress()

    assert gate.hosting_mode == LOCAL
    assert not gate.is_hosted
    assert gate.allowed
    assert not gate.content_leaves_machine
    assert not gate.blocked
    assert gate.warnings == ()


def test_a_caller_who_passes_nothing_cannot_send_anything(no_network: None) -> None:
    # Every default is the safe one, so an incomplete call is a safe call.
    assert not provider_data_preflight().content_leaves_machine


def test_the_default_classification_is_unknown_not_public(no_network: None) -> None:
    # public is the one classification that silences the unredacted warning, so
    # defaulting to it would assert something nobody said and hide the warning.
    assert DEFAULT_CLASSIFICATION == "unknown"
    assert provider_data_preflight().dataset_privacy_classification == "unknown"


def test_a_local_run_writes_an_egress_block_saying_nothing_left(source: Path, tmp_path: Path, no_network: None) -> None:
    outcome = _run(source, tmp_path / "target")

    egress = json.loads(outcome.record.read_text(encoding="utf-8"))["egress"]
    assert egress["remote_provider_used"] is False
    assert egress["records_transmitted"] == 0
    assert egress["raw_source_text_transmitted"] is False
    assert egress["hosting_mode"] == "local"
    assert egress["hosted_execution_confirmed"] is False
    # The block is always written, so its absence can never be mistaken for a
    # local run.
    assert "warnings" not in egress


# --- negative test 6: a remote provider without the opt-in ------------------


def test_a_hosted_run_without_the_opt_in_is_refused_before_anything_is_read(source: Path, tmp_path: Path, no_network: None) -> None:
    gate = provider_data_preflight(provider="example-hosted", hosting_mode=HOSTED, allow_hosted=False)
    assert gate.blocked

    with pytest.raises(ReEmbedError, match="Hosted execution was not enabled"):
        _run(source, tmp_path / "target", egress=gate)

    # Nothing was written, so nothing was read far enough to write it.
    assert not (tmp_path / "target" / "re-embed-record.json").exists()
    assert not (tmp_path / "target" / "vectors").exists()


def test_the_refusal_does_not_depend_on_the_input_being_readable(tmp_path: Path, no_network: None) -> None:
    # The gate closes before the evidence document is opened, so a hosted run
    # is refused for the right reason even when the input is nonsense.
    absent = tmp_path / "nothing-here.json"
    gate = provider_data_preflight(provider="example-hosted", hosting_mode=HOSTED, allow_hosted=False)

    with pytest.raises(ReEmbedError, match="Hosted execution was not enabled"):
        _run(absent, tmp_path / "target", egress=gate)


def test_the_refusal_names_the_warning_the_operator_would_have_seen(source: Path, tmp_path: Path, no_network: None) -> None:
    gate = provider_data_preflight(provider="example-hosted", hosting_mode=HOSTED, allow_hosted=False)

    with pytest.raises(ReEmbedError, match=BLOCKED_WARNING):
        _run(source, tmp_path / "target", egress=gate)


def test_a_blocked_run_cannot_smuggle_an_embedder_past_the_gate(source: Path, tmp_path: Path, no_network: None) -> None:
    gate = provider_data_preflight(provider="example-hosted", hosting_mode=HOSTED, allow_hosted=False)

    with pytest.raises(ReEmbedError, match="Hosted execution was not enabled"):
        _run(source, tmp_path / "target", egress=gate, embedder=build_built_in_embedder(MODEL))


# --- the opt-in -------------------------------------------------------------


def test_an_allowed_hosted_run_needs_an_embedder_the_operator_supplies(source: Path, tmp_path: Path, no_network: None) -> None:
    # This package ships no hosted provider. Falling back to a local model
    # would make the record say hosted about a run that was not.
    gate = provider_data_preflight(provider="example-hosted", hosting_mode=HOSTED, allow_hosted=True)

    with pytest.raises(ReEmbedError, match="ships no hosted provider"):
        _run(source, tmp_path / "target", egress=gate)


def test_an_allowed_hosted_run_reports_what_left_the_machine(source: Path, tmp_path: Path, no_network: None) -> None:
    # The embedder stands in for an operator's hosted client. It is local and
    # deterministic, which is what lets this assert the reporting without a
    # network call.
    gate = provider_data_preflight(
        provider="example-hosted",
        hosting_mode=HOSTED,
        dataset_privacy_classification="internal",
        allow_hosted=True,
    )

    outcome = _run(source, tmp_path / "target", egress=gate, embedder=build_built_in_embedder(MODEL))

    egress = json.loads(outcome.record.read_text(encoding="utf-8"))["egress"]
    assert egress["remote_provider_used"] is True
    assert egress["provider"] == "example-hosted"
    assert egress["records_transmitted"] == outcome.retained_rebuilt
    assert egress["raw_source_text_transmitted"] is True
    assert egress["hosted_execution_confirmed"] is True
    assert egress["dataset_privacy_classification"] == "internal"
    assert UNREDACTED_WARNING in egress["warnings"]


def test_only_retained_records_are_counted_as_transmitted(tmp_path: Path, no_network: None) -> None:
    # Records with no source text have nothing to send, so counting them would
    # over-report what left the machine.
    source = build_source_store(tmp_path / "evidence", dimensions=8, missing=("gone",), deleted_by_policy=("purged",))
    gate = provider_data_preflight(provider="example-hosted", hosting_mode=HOSTED, allow_hosted=True)

    outcome = _run(source, tmp_path / "target", egress=gate, embedder=build_built_in_embedder(MODEL))

    egress = json.loads(outcome.record.read_text(encoding="utf-8"))["egress"]
    assert egress["records_transmitted"] == 3
    assert outcome.missing_skipped == 1
    assert outcome.deleted_by_policy_skipped == 1


def test_a_redacted_hosted_run_says_the_raw_text_did_not_go(source: Path, tmp_path: Path, no_network: None) -> None:
    gate = provider_data_preflight(
        provider="example-hosted",
        hosting_mode=HOSTED,
        dataset_privacy_classification="confidential",
        allow_hosted=True,
        redaction_applied=True,
    )

    outcome = _run(source, tmp_path / "target", egress=gate, embedder=build_built_in_embedder(MODEL))

    egress = json.loads(outcome.record.read_text(encoding="utf-8"))["egress"]
    assert egress["raw_source_text_transmitted"] is False
    assert egress["redaction_applied"] is True
    # A claimed redaction silences the warning, which is why the claim itself
    # is recorded and labelled attested rather than taken as a fact.
    assert "warnings" not in egress


# --- the warnings -----------------------------------------------------------


def test_the_blocked_warning_appears_only_when_hosted_and_not_allowed(no_network: None) -> None:
    blocked = provider_data_preflight(provider="p", hosting_mode=HOSTED, allow_hosted=False)
    allowed = provider_data_preflight(provider="p", hosting_mode=HOSTED, allow_hosted=True)

    assert BLOCKED_WARNING in blocked.warnings
    assert BLOCKED_WARNING not in allowed.warnings
    assert BLOCKED_WARNING not in local_egress().warnings


@pytest.mark.parametrize("classification", ("internal", "confidential", "restricted"))
def test_private_content_raises_the_unredacted_warning(classification: str, no_network: None) -> None:
    gate = provider_data_preflight(provider="p", hosting_mode=HOSTED, dataset_privacy_classification=classification, allow_hosted=True)

    assert UNREDACTED_WARNING in gate.warnings


@pytest.mark.parametrize("classification", ("public", "unknown"))
def test_open_content_does_not_raise_the_unredacted_warning(classification: str, no_network: None) -> None:
    gate = provider_data_preflight(provider="p", hosting_mode=HOSTED, dataset_privacy_classification=classification, allow_hosted=True)

    assert UNREDACTED_WARNING not in gate.warnings


def test_telemetry_raises_its_warning_even_on_a_local_run(no_network: None) -> None:
    gate = provider_data_preflight(telemetry_enabled=True)

    assert TELEMETRY_WARNING in gate.warnings
    assert not gate.content_leaves_machine


def test_the_warning_wordings_match_the_benchmarking_tool(no_network: None) -> None:
    # PrivateAIStack and ModelSwapBench take no dependency on each other, so
    # these sentences are restated rather than imported. An operator who has
    # read one tool's warning should not have to learn a second wording for the
    # same risk, and this test is where a drift between them shows.
    assert BLOCKED_WARNING == "hosted execution is blocked until explicitly enabled"
    assert UNREDACTED_WARNING == ("private unredacted dataset content would be transmitted to a hosted provider")
    assert TELEMETRY_WARNING == "telemetry is enabled; review exported fields before execution"


# --- refused settings -------------------------------------------------------


def test_a_hosted_run_must_name_its_provider(no_network: None) -> None:
    with pytest.raises(EgressError, match="must name its provider"):
        provider_data_preflight(hosting_mode=HOSTED, allow_hosted=True)


def test_a_local_run_must_not_name_a_provider(no_network: None) -> None:
    # Either a mistake, or a claim the record would carry with nothing behind it.
    with pytest.raises(EgressError, match="must not name a provider"):
        provider_data_preflight(provider="example-hosted", hosting_mode=LOCAL)


@pytest.mark.parametrize("mode", ("serverless", "", "HOSTED"))
def test_an_unknown_hosting_mode_is_refused(mode: str, no_network: None) -> None:
    with pytest.raises(EgressError, match="hosting_mode must be one of"):
        provider_data_preflight(hosting_mode=mode)


def test_an_unknown_classification_is_refused(no_network: None) -> None:
    with pytest.raises(EgressError, match="dataset_privacy_classification must be one of"):
        provider_data_preflight(dataset_privacy_classification="top-secret")


def test_an_oversized_provider_name_is_refused(no_network: None) -> None:
    with pytest.raises(EgressError, match="must name its provider"):
        provider_data_preflight(provider="x" * 129, hosting_mode=HOSTED, allow_hosted=True)


def test_a_run_that_sent_nothing_cannot_report_transmitted_records(no_network: None) -> None:
    with pytest.raises(EgressError, match="cannot report transmitted records"):
        local_egress().to_document(records_transmitted=5)


def test_a_negative_transmitted_count_is_refused(no_network: None) -> None:
    with pytest.raises(EgressError, match="cannot be negative"):
        local_egress().to_document(records_transmitted=-1)


def test_a_hosted_run_that_sent_zero_records_reports_no_provider(no_network: None) -> None:
    # Allowed, but nothing went, so the record must not claim a remote run.
    # The suite's reader refuses a local-looking run that names a provider.
    gate = provider_data_preflight(provider="example-hosted", hosting_mode=HOSTED, allow_hosted=True)

    document = gate.to_document(records_transmitted=0)

    assert document["remote_provider_used"] is False
    assert "provider" not in document
    assert document["hosting_mode"] == "hosted"
    assert document["hosted_execution_confirmed"] is True
