"""Trust-boundary tests for built-in and injected re-embedding executors."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from private_ai_stack.reembed import (
    DEFAULT_CLASSIFICATION,
    EXTERNAL_UNVERIFIED,
    HOSTED,
    LOCAL,
    LOCAL_BUILTIN,
    EgressError,
    ModelIdentity,
    ReEmbedError,
    provider_data_preflight,
    re_embed_store,
)
from private_ai_stack.reembed.embedders import Embedder, build_built_in_embedder
from private_ai_stack.reembed.privacy import (
    BLOCKED_WARNING,
    CLASSIFICATION_BLOCKED_WARNING,
    EXTERNAL_ATTESTED_WARNING,
    HOSTED_BLOCKED_WARNING,
    TELEMETRY_WARNING,
    local_egress,
)
from tests.unit.reembed_support import build_source_store, tree_digest

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


def _external_local(**overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "executor_kind": EXTERNAL_UNVERIFIED,
        "provider": "local-test-executor",
        "endpoint": "http://127.0.0.1:11434/embed",
        "allow_external": True,
    }
    arguments.update(overrides)
    return provider_data_preflight(**arguments)


def _external_hosted(**overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "executor_kind": EXTERNAL_UNVERIFIED,
        "provider": "example-hosted",
        "endpoint": "https://embeddings.example/v1",
        "hosting_mode": HOSTED,
        "dataset_privacy_classification": "public",
        "allow_external": True,
        "allow_hosted": True,
    }
    arguments.update(overrides)
    return provider_data_preflight(**arguments)


class RecordingEmbedder:
    def __init__(self) -> None:
        self._inner = build_built_in_embedder(MODEL)
        self.texts: list[str] = []

    @property
    def identity(self) -> ModelIdentity:
        return self._inner.identity

    def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return self._inner.embed(text)


class InterruptingEmbedder(RecordingEmbedder):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def embed(self, text: str) -> list[float]:
        if len(self.texts) >= self.limit:
            raise KeyboardInterrupt("simulated interruption")
        return super().embed(text)


def test_default_is_registry_owned_local_execution(source: Path, tmp_path: Path) -> None:
    gate = local_egress()
    assert gate.executor_kind == LOCAL_BUILTIN
    assert gate.hosting_mode == LOCAL
    assert gate.allowed
    assert not gate.is_external
    assert not gate.content_leaves_machine

    outcome = _run(source, tmp_path / "target")
    egress = json.loads(outcome.record.read_text(encoding="utf-8"))["egress"]
    assert egress["executor_kind"] == LOCAL_BUILTIN
    assert egress["execution_verification"] == "registry_owned"
    assert egress["records_submitted_to_external_executor"] == 0


def test_old_local_call_and_explicit_local_gate_are_byte_compatible(source: Path, tmp_path: Path) -> None:
    implicit = tmp_path / "implicit"
    explicit = tmp_path / "explicit"
    _run(source, implicit)
    _run(source, explicit, egress=local_egress())
    assert tree_digest(implicit) == tree_digest(explicit)


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_external_opt_in_must_be_an_exact_boolean(value: object) -> None:
    with pytest.raises(EgressError, match="allow_external must be a boolean"):
        provider_data_preflight(
            executor_kind=EXTERNAL_UNVERIFIED,
            provider="external",
            endpoint="http://127.0.0.1:11434",
            allow_external=value,  # type: ignore[arg-type]
        )


def test_omitted_external_opt_in_is_default_deny() -> None:
    gate = provider_data_preflight(
        executor_kind=EXTERNAL_UNVERIFIED,
        provider="external",
        endpoint="http://127.0.0.1:11434",
    )
    assert gate.blocked
    assert BLOCKED_WARNING in gate.warnings


def test_false_external_opt_in_blocks_before_an_absent_source_is_read(tmp_path: Path) -> None:
    gate = provider_data_preflight(
        executor_kind=EXTERNAL_UNVERIFIED,
        provider="external",
        endpoint="http://127.0.0.1:11434",
        allow_external=False,
    )
    assert gate.blocked
    with pytest.raises(ReEmbedError, match="refused before anything is read"):
        _run(tmp_path / "does-not-exist.json", tmp_path / "target", embedder=RecordingEmbedder(), egress=gate)
    assert not (tmp_path / "target").exists()


def test_injected_embedder_without_external_preflight_is_refused_before_read(tmp_path: Path) -> None:
    with pytest.raises(ReEmbedError, match="requires explicit external consent"):
        _run(tmp_path / "does-not-exist.json", tmp_path / "target", embedder=RecordingEmbedder())


@pytest.mark.parametrize(
    "change",
    [
        {"allowed": False},
        {"content_leaves_machine": True},
        {"external_execution_confirmed": True},
        {"warnings": (BLOCKED_WARNING,)},
    ],
)
def test_direct_preflight_construction_rejects_contradictions(change: dict[str, object]) -> None:
    with pytest.raises(EgressError):
        replace(local_egress(), **change)


def test_hosted_execution_requires_both_consents() -> None:
    external_missing = provider_data_preflight(
        executor_kind=EXTERNAL_UNVERIFIED,
        provider="hosted",
        endpoint="https://embeddings.example",
        hosting_mode=HOSTED,
        dataset_privacy_classification="public",
        allow_external=False,
        allow_hosted=True,
    )
    hosted_missing = provider_data_preflight(
        executor_kind=EXTERNAL_UNVERIFIED,
        provider="hosted",
        endpoint="https://embeddings.example",
        hosting_mode=HOSTED,
        dataset_privacy_classification="public",
        allow_external=True,
        allow_hosted=False,
    )
    assert external_missing.blocked and BLOCKED_WARNING in external_missing.warnings
    assert hosted_missing.blocked and HOSTED_BLOCKED_WARNING in hosted_missing.warnings


@pytest.mark.parametrize("classification", ["unknown", "confidential", "restricted"])
def test_sensitive_or_unknown_data_is_blocked_from_hosted_execution(classification: str) -> None:
    gate = _external_hosted(dataset_privacy_classification=classification)
    assert gate.blocked
    assert CLASSIFICATION_BLOCKED_WARNING in gate.warnings


@pytest.mark.parametrize("classification", ["public", "internal"])
def test_permitted_classifications_still_require_explicit_hosted_consent(classification: str) -> None:
    assert _external_hosted(dataset_privacy_classification=classification).allowed


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://embeddings.example/v1",
        "https://localhost/v1",
        "https://127.0.0.1/v1",
        "https://10.0.0.8/v1",
        "https://user:secret@embeddings.example/v1",
        "https://embeddings.example/v1?token=secret",
        "https://embeddings.example/v1#fragment",
    ],
)
def test_unsafe_hosted_endpoint_claims_are_refused_without_echoing_secrets(endpoint: str) -> None:
    with pytest.raises(EgressError) as failure:
        _external_hosted(endpoint=endpoint)
    assert "secret" not in str(failure.value)


def test_local_external_endpoint_must_be_loopback() -> None:
    with pytest.raises(EgressError, match="loopback"):
        _external_local(endpoint="https://embeddings.example/v1")


def test_proxy_and_redirect_claims_are_refused() -> None:
    with pytest.raises(EgressError, match="Proxy routing"):
        _external_hosted(proxy_url="https://proxy.example")
    with pytest.raises(EgressError, match="Redirect following"):
        _external_hosted(allow_redirects=True)


def test_untransformed_source_text_cannot_claim_redaction() -> None:
    with pytest.raises(EgressError, match="Redaction is not implemented"):
        _external_hosted(redaction_applied=True)


def test_hosted_record_derives_raw_transmission_from_submitted_payloads(source: Path, tmp_path: Path) -> None:
    outcome = _run(
        source,
        tmp_path / "target",
        embedder=build_built_in_embedder(MODEL),
        egress=_external_hosted(),
    )
    egress = json.loads(outcome.record.read_text(encoding="utf-8"))["egress"]
    assert egress["execution_verification"] == "attested"
    assert egress["network_behavior_verification"] == "attested"
    assert egress["records_submitted_to_external_executor"] == 3
    assert egress["records_transmitted"] == 3
    assert egress["raw_source_text_transmitted"] is True
    assert egress["redaction_applied"] is False


def test_local_external_execution_is_attested_not_remote(source: Path, tmp_path: Path) -> None:
    outcome = _run(source, tmp_path / "target", embedder=RecordingEmbedder(), egress=_external_local())
    egress = json.loads(outcome.record.read_text(encoding="utf-8"))["egress"]
    assert egress["records_submitted_to_external_executor"] == 3
    assert egress["records_transmitted"] == 0
    assert egress["remote_provider_used"] is False
    assert egress["network_behavior_verification"] == "attested"
    assert EXTERNAL_ATTESTED_WARNING in egress["warnings"]


def test_missing_and_deleted_records_never_reach_the_external_executor(tmp_path: Path) -> None:
    source = build_source_store(
        tmp_path / "evidence",
        texts={"kept": "retained text"},
        missing=("missing-secret",),
        deleted_by_policy=("deleted-secret",),
        dimensions=8,
    )
    recorder = RecordingEmbedder()
    outcome = _run(source, tmp_path / "target", embedder=recorder, egress=_external_local())
    assert recorder.texts == ["retained text"]
    egress = json.loads(outcome.record.read_text(encoding="utf-8"))["egress"]
    assert egress["records_submitted_to_external_executor"] == 1
    assert outcome.missing_skipped == outcome.deleted_by_policy_skipped == 1


def test_source_text_and_secrets_are_absent_from_executor_diagnostics(source: Path, tmp_path: Path) -> None:
    class LeakingFailure(RecordingEmbedder):
        def embed(self, text: str) -> list[float]:
            raise RuntimeError(f"provider token super-secret; source={text}")

    with pytest.raises(ReEmbedError) as failure:
        _run(source, tmp_path / "target", embedder=LeakingFailure(), egress=_external_local())
    message = str(failure.value)
    assert "super-secret" not in message
    assert "Reset the password" not in message
    assert message == "The external executor failed while processing a source record."


def test_resume_keeps_distinct_external_submission_count_consistent(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "evidence", texts={f"r{i}": f"text {i}" for i in range(5)}, dimensions=8)
    target = tmp_path / "target"
    gate = _external_local()
    with pytest.raises(KeyboardInterrupt):
        _run(source, target, batch_size=2, embedder=InterruptingEmbedder(2), egress=gate)

    resumed = _run(source, target, batch_size=2, embedder=RecordingEmbedder(), egress=gate)
    egress = json.loads(resumed.record.read_text(encoding="utf-8"))["egress"]
    assert resumed.resumed
    assert egress["records_submitted_to_external_executor"] == 5
    assert egress["records_transmitted"] == 0


def test_resume_cannot_switch_external_executor_context(tmp_path: Path) -> None:
    source = build_source_store(tmp_path / "evidence", texts={f"r{i}": f"text {i}" for i in range(4)}, dimensions=8)
    target = tmp_path / "target"
    with pytest.raises(KeyboardInterrupt):
        _run(source, target, batch_size=2, embedder=InterruptingEmbedder(2), egress=_external_local())
    changed = _external_local(provider="other-local-executor")
    with pytest.raises(ReEmbedError, match="different executor or consent context"):
        _run(source, target, batch_size=2, embedder=RecordingEmbedder(), egress=changed)


def test_local_builtin_cannot_report_external_payloads() -> None:
    with pytest.raises(EgressError, match="cannot report external payloads"):
        local_egress().to_document(records_submitted=1)


def test_default_classification_is_unknown_without_making_hosted_unknown_safe() -> None:
    assert DEFAULT_CLASSIFICATION == "unknown"
    assert local_egress().dataset_privacy_classification == "unknown"
    assert _external_hosted(dataset_privacy_classification="unknown").blocked


def test_telemetry_is_attested_and_warned_for_external_execution() -> None:
    gate = _external_local(telemetry_enabled=True)
    assert TELEMETRY_WARNING in gate.warnings
    assert gate.telemetry_enabled


def test_embedder_protocol_remains_structurally_compatible() -> None:
    assert isinstance(RecordingEmbedder(), Embedder)
