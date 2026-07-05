from fastapi import APIRouter, Depends, Request

from private_ai_stack.api.dependencies import task_service
from private_ai_stack.api.schemas import TaskRequest, TaskResponse
from private_ai_stack.services.task_service import TaskService

router = APIRouter(tags=["tasks"])


@router.post("/tasks", response_model=TaskResponse)
async def create_task(payload: TaskRequest, request: Request, service: TaskService = Depends(task_service)) -> TaskResponse:
    return await service.create_task(payload, request.state.request_id, getattr(request.state, "trace_id", None))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, service: TaskService = Depends(task_service)) -> TaskResponse:
    return service.get_task(task_id)


@router.get("/tasks/{task_id}/events")
async def get_task_events(task_id: str, service: TaskService = Depends(task_service)) -> dict[str, object]:
    return {"task_id": task_id, "events": service.get_events(task_id)}


@router.get("/tasks/{task_id}/audit")
async def get_task_audit(task_id: str, service: TaskService = Depends(task_service)) -> dict[str, object]:
    return {"task_id": task_id, "audit": service.get_audit(task_id)}
