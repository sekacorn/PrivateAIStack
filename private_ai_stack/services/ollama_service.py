from typing import Any

import httpx

from private_ai_stack.config.settings import Settings


class OllamaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def status(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                response.raise_for_status()
            return "ok"
        except Exception:
            return "unavailable"

    async def models(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.settings.ollama_base_url}/api/tags")
                response.raise_for_status()
            payload = response.json()
            names = [item.get("name") for item in payload.get("models", [])]
            return {"status": "ok", "models": names, "configured_model_present": self.settings.ollama_model in names}
        except Exception as exc:
            return {"status": "unavailable", "error": exc.__class__.__name__, "models": [], "configured_model_present": False}

    async def ensure_model(self) -> tuple[bool, str]:
        info = await self.models()
        if info["status"] != "ok":
            return False, "ollama_unavailable"
        if not info["configured_model_present"]:
            return False, "model_missing"
        return True, "ok"

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self.settings.ollama_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json={"model": self.settings.ollama_model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return str(response.json().get("response", "")).strip()
