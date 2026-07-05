from pathlib import Path

import pytest

from private_ai_stack.api.schemas import TaskRequest
from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.config.settings import Settings
from private_ai_stack.services.task_service import TaskService


class FakeForge:
    async def run_plan_task(self, goal: str, task_id: str, request_id: str, trace_id: str | None, actor: str) -> dict[str, object]:
        return {"output": goal, "task_id": task_id, "actor": actor}


class FailingForge:
    async def run_plan_task(self, goal: str, task_id: str, request_id: str, trace_id: str | None, actor: str) -> dict[str, object]:
        raise RuntimeError("model_missing")


@pytest.mark.asyncio
async def test_task_service_success_and_lookup(tmp_path: Path) -> None:
    settings = Settings(
        database_url="memory://local",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
    )
    service = TaskService(FakeForge(), AuditWriter(tmp_path / "audit.jsonl"), settings)  # type: ignore[arg-type]

    task = await service.create_task(TaskRequest(goal="plan it"), "req-1", None)

    assert task.status == "succeeded"
    assert service.get_task(task.task_id).result
    assert service.get_events(task.task_id)
    assert service.get_audit(task.task_id)


@pytest.mark.asyncio
async def test_task_service_failure(tmp_path: Path) -> None:
    settings = Settings(
        database_url="memory://local",
        audit_dir=tmp_path / "audit",
        reports_dir=tmp_path / "reports",
        export_dir=tmp_path / "exports",
    )
    service = TaskService(FailingForge(), AuditWriter(tmp_path / "audit.jsonl"), settings)  # type: ignore[arg-type]

    task = await service.create_task(TaskRequest(goal="plan it"), "req-1", None)

    assert task.status == "failed"
    assert task.error == "model_missing"
