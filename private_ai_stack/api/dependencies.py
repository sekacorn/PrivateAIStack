from fastapi import Header, Request

from private_ai_stack.api.errors import AppError
from private_ai_stack.config.settings import Settings, get_settings
from private_ai_stack.services.knowledge_service import KnowledgeService
from private_ai_stack.services.review_service import ReviewService
from private_ai_stack.services.task_service import TaskService


def settings_dependency() -> Settings:
    return get_settings()


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    if settings.api_key and x_api_key != settings.api_key:
        raise AppError("unauthorized", "Invalid or missing API key.", 401)


def task_service(request: Request) -> TaskService:
    service: TaskService = request.app.state.task_service
    return service


def knowledge_service(request: Request) -> KnowledgeService:
    service: KnowledgeService = request.app.state.knowledge_service
    return service


def review_service(request: Request) -> ReviewService:
    service: ReviewService = request.app.state.review_service
    return service
