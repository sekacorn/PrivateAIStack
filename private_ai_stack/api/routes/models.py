from fastapi import APIRouter, Request

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> dict[str, object]:
    ollama = request.app.state.ollama
    return {
        "default": ollama.settings.ollama_model,
        "provider": "ollama",
        "hosted_providers_enabled": ollama.settings.allow_hosted_providers,
        "ollama": await ollama.models(),
    }
