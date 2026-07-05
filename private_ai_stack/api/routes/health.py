import importlib.metadata as md

from fastapi import APIRouter, Request

from private_ai_stack import __version__
from private_ai_stack.api.schemas import HealthResponse, ReadyResponse, VersionResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    model_ready, model_reason = await request.app.state.ollama.ensure_model()
    checks = {
        "ollama": await request.app.state.ollama.status(),
        "ollama_model": "ok" if model_ready else model_reason,
        "memory": await request.app.state.memory.status(),
    }
    overall = "ready" if all(value == "ok" for value in checks.values()) else "degraded"
    return ReadyResponse(status=overall, checks=checks)


@router.get("/version", response_model=VersionResponse)
async def version(request: Request) -> VersionResponse:
    return VersionResponse(
        version=__version__,
        forge_version=md.version("agentforge-oss"),
        default_model=request.app.state.ollama.settings.ollama_model,
    )
