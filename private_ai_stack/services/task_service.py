import uuid

from private_ai_stack.api.errors import AppError
from private_ai_stack.api.schemas import Status, TaskRequest, TaskResponse, utc_now
from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.config.settings import Settings
from private_ai_stack.observability.telemetry import span
from private_ai_stack.services.forge_service import ForgeService


class TaskService:
    def __init__(self, forge: ForgeService, audit: AuditWriter, settings: Settings) -> None:
        self.forge = forge
        self.audit = audit
        self.settings = settings
        self._tasks: dict[str, TaskResponse] = {}
        self._events: dict[str, list[dict[str, object]]] = {}

    async def create_task(self, payload: TaskRequest, request_id: str, trace_id: str | None) -> TaskResponse:
        with span("task.create"):
            task_id = str(uuid.uuid4())
            now = utc_now()
            task = TaskResponse(
                task_id=task_id, status=Status.running, request_id=request_id, trace_id=trace_id, created_at=now, updated_at=now
            )
            self._tasks[task_id] = task
            self._events[task_id] = [{"event": "task.created", "timestamp": now.isoformat()}]
            self.audit.write(
                "task.created",
                entity_type="task",
                entity_id=task_id,
                actor=payload.actor,
                request_id=request_id,
                trace_id=trace_id,
                details={"goal_length": len(payload.goal)},
            )
            try:
                result = await self.forge.run_plan_task(payload.goal, task_id, request_id, trace_id, payload.actor)
                task.status = Status.succeeded
                task.result = result
                self._events[task_id].append({"event": "task.succeeded", "timestamp": utc_now().isoformat()})
                self.audit.write(
                    "task.succeeded",
                    entity_type="task",
                    entity_id=task_id,
                    actor=payload.actor,
                    request_id=request_id,
                    trace_id=trace_id,
                    details={"result_length": len(str(result.get("output", "")))},
                )
            except RuntimeError as exc:
                task.status = Status.failed
                task.error = str(exc)
                self._events[task_id].append({"event": "task.failed", "timestamp": utc_now().isoformat(), "error": str(exc)})
                self.audit.write(
                    "task.failed",
                    entity_type="task",
                    entity_id=task_id,
                    actor=payload.actor,
                    request_id=request_id,
                    trace_id=trace_id,
                    details={"error": str(exc)},
                )
            task.updated_at = utc_now()
            return task

    def get_task(self, task_id: str) -> TaskResponse:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise AppError("not_found", "Task not found.", 404, {"task_id": task_id}) from exc

    def get_events(self, task_id: str) -> list[dict[str, object]]:
        self.get_task(task_id)
        return self._events.get(task_id, [])

    def get_audit(self, task_id: str) -> list[dict[str, object]]:
        self.get_task(task_id)
        return self.audit.records_for(task_id)
