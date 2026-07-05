import asyncio
import importlib.metadata as md
import inspect
from typing import Any

from forge import BudgetConfig, ForgeConfig, ModelInfo, ModelRegistry, ModelTier, OllamaProvider, Orchestrator, RoutingConfig
from pydantic import SecretStr

from private_ai_stack.audit.writer import AuditWriter
from private_ai_stack.config.settings import Settings
from private_ai_stack.policies.defaults import build_policy_set
from private_ai_stack.services.ollama_service import OllamaService


class ForgeService:
    """Runs plan tasks through Forge while preserving a local-only fallback path."""

    def __init__(self, settings: Settings, audit: AuditWriter, ollama: OllamaService) -> None:
        self.settings = settings
        self.audit = audit
        self.ollama = ollama
        self.version = md.version("agentforge-oss")

    async def run_plan_task(self, goal: str, task_id: str, request_id: str, trace_id: str | None, actor: str) -> dict[str, Any]:
        """Use Forge first, then fall back to direct Ollama without changing provider boundaries."""
        ready, reason = await self.ollama.ensure_model()
        if not ready:
            raise RuntimeError(reason)

        self.audit.write(
            "task.model_selected",
            entity_type="task",
            entity_id=task_id,
            actor=actor,
            request_id=request_id,
            trace_id=trace_id,
            details={"provider": "ollama", "model": self.settings.ollama_model},
        )
        prompt = (
            "Analyze the supplied goal and return a concise implementation plan with risks, assumptions, and next actions. "
            "Do not claim execution. Keep it practical.\n\nGoal:\n"
            f"{goal}"
        )

        async def _run_forge() -> dict[str, Any]:
            # Fixed routing prevents Forge from selecting a hosted provider even if one is installed globally.
            config = ForgeConfig(
                routing=RoutingConfig(
                    strategy="fixed",
                    default_model=self.settings.ollama_model,
                    default_provider="ollama",
                    allow_providers=["ollama"],
                ),
                budget=BudgetConfig(max_workers=self.settings.max_workers, max_steps_per_agent=self.settings.max_agent_steps),
                ollama_base_url=SecretStr(self.settings.ollama_base_url),
                memory_backend="pgvector" if self.settings.database_url.startswith("postgresql") else "inmemory",
                pgvector_dsn=self.settings.database_url if self.settings.database_url.startswith("postgresql") else None,
                otel_enabled=self.settings.otel_enabled,
                otel_endpoint=self.settings.otel_endpoint,
                otel_service_name=self.settings.otel_service_name,
            )
            provider = OllamaProvider(base_url=self.settings.ollama_base_url, timeout=self.settings.ollama_timeout_seconds)
            registry = ModelRegistry()
            # Register the configured model explicitly so Forge sees the same local model as readiness checks.
            registry.register(
                ModelInfo(
                    name=self.settings.ollama_model,
                    provider="ollama",
                    tier=ModelTier.SMALL,
                    context_window=32768,
                    max_output_tokens=4096,
                    input_cost_per_mtok=0.0,
                    output_cost_per_mtok=0.0,
                    supports_tools=True,
                    description="Local Ollama model configured by PrivateAIStack.",
                )
            )
            orchestrator = Orchestrator(config=config, providers={"ollama": provider}, registry=registry)
            run_result: Any = orchestrator.run(prompt, mode="single", model=self.settings.ollama_model, policy_set=build_policy_set())
            if inspect.isawaitable(run_result):
                run_result = await run_result
            return {
                "output": getattr(run_result, "output", str(run_result)),
                "usage": str(getattr(run_result, "usage", "")),
                "provider": "ollama",
                "model": self.settings.ollama_model,
                "forge_version": self.version,
            }

        try:
            return await asyncio.wait_for(_run_forge(), timeout=self.settings.task_timeout_seconds)
        except Exception as exc:
            # Narrow runtime fallback: still use configured local Ollama, never a hosted model.
            text = await asyncio.wait_for(self.ollama.generate(prompt), timeout=self.settings.task_timeout_seconds)
            self.audit.write(
                "task.forge_fallback",
                entity_type="task",
                entity_id=task_id,
                actor=actor,
                request_id=request_id,
                trace_id=trace_id,
                details={"reason": exc.__class__.__name__, "provider": "ollama"},
            )
            return {
                "output": text,
                "usage": "ollama-direct-fallback",
                "fallback_reason": exc.__class__.__name__,
                "provider": "ollama",
                "model": self.settings.ollama_model,
                "forge_version": self.version,
            }
