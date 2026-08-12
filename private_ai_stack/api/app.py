from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from private_ai_stack import __version__
from private_ai_stack.api.dependencies import require_api_key
from private_ai_stack.api.errors import register_error_handlers
from private_ai_stack.api.middleware import request_context_middleware
from private_ai_stack.api.routes import health, knowledge, models, policies, reviews, tasks
from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.config.settings import get_settings
from private_ai_stack.memory.store import MemoryStore
from private_ai_stack.observability.telemetry import configure_telemetry
from private_ai_stack.services.forge_service import ForgeService
from private_ai_stack.services.knowledge_service import KnowledgeService
from private_ai_stack.services.ollama_service import OllamaService
from private_ai_stack.services.review_service import ReviewService
from private_ai_stack.services.task_service import TaskService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_telemetry(settings)
    audit = AuditWriter(settings.audit_dir / "audit.jsonl", max_record_bytes=settings.max_audit_record_bytes)
    memory = MemoryStore(settings)
    # PostgreSQL is the durable default. Startup fails instead of silently losing RAG data.
    await memory.initialize()
    ollama = OllamaService(settings)
    forge = ForgeService(settings, audit, ollama)
    app.state.audit = audit
    app.state.memory = memory
    app.state.ollama = ollama
    app.state.forge = forge
    app.state.task_service = TaskService(forge, audit, settings)
    app.state.knowledge_service = KnowledgeService(memory, audit)
    app.state.review_service = ReviewService(settings, audit)
    yield
    await memory.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="PrivateAIStack API",
        version=__version__,
        description="Local-first AI agents, persistent RAG, governed code review, auditability, and optional observability.",
        lifespan=lifespan,
    )
    app.middleware("http")(request_context_middleware)
    register_error_handlers(app)
    app.include_router(health.router)
    protected = [Depends(require_api_key)]
    app.include_router(tasks.router, prefix="/v1", dependencies=protected)
    app.include_router(knowledge.router, prefix="/v1", dependencies=protected)
    app.include_router(reviews.router, prefix="/v1", dependencies=protected)
    app.include_router(models.router, prefix="/v1", dependencies=protected)
    app.include_router(policies.router, prefix="/v1", dependencies=protected)
    return app


app = create_app()
