from pathlib import Path

import pytest

from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.config.settings import Settings
from private_ai_stack.services import forge_service
from private_ai_stack.services.forge_service import ForgeService


class FakeOllama:
    async def ensure_model(self) -> tuple[bool, str]:
        return True, "ok"

    async def generate(self, prompt: str) -> str:
        return f"direct: {prompt[:12]}"


class FailingFallbackOllama(FakeOllama):
    async def generate(self, prompt: str) -> str:
        raise ValueError("malformed_response")


class FailingOrchestrator:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def run(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("forge failed")


def settings(tmp_path: Path, *, allow_fallback: bool) -> Settings:
    return Settings(
        database_url="memory://local",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
        allow_direct_ollama_fallback=allow_fallback,
    )


@pytest.mark.asyncio
async def test_forge_failure_is_fail_closed_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forge_service, "Orchestrator", FailingOrchestrator)
    audit = AuditWriter(tmp_path / "audit.jsonl")
    service = ForgeService(settings(tmp_path, allow_fallback=False), audit, FakeOllama())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="forge_execution_failed"):
        await service.run_plan_task("plan safely", "task-1", "request-1", None, "local-user")

    record = audit.records_for("task-1")[-1]
    assert record["event_type"] == "task.forge_failed"
    assert record["details"]["fallback"] == "disabled"


@pytest.mark.asyncio
async def test_direct_ollama_fallback_requires_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forge_service, "Orchestrator", FailingOrchestrator)
    audit = AuditWriter(tmp_path / "audit.jsonl")
    service = ForgeService(settings(tmp_path, allow_fallback=True), audit, FakeOllama())  # type: ignore[arg-type]

    result = await service.run_plan_task("plan safely", "task-1", "request-1", None, "local-user")

    assert result["usage"] == "ollama-direct-fallback"
    assert audit.records_for("task-1")[-1]["event_type"] == "task.forge_fallback"


@pytest.mark.asyncio
async def test_opted_in_fallback_still_has_consistent_failure_semantics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forge_service, "Orchestrator", FailingOrchestrator)
    audit = AuditWriter(tmp_path / "audit.jsonl")
    service = ForgeService(settings(tmp_path, allow_fallback=True), audit, FailingFallbackOllama())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="ollama_fallback_failed"):
        await service.run_plan_task("plan safely", "task-1", "request-1", None, "local-user")

    assert audit.records_for("task-1")[-1]["event_type"] == "task.forge_fallback_failed"
