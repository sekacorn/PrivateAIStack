from collections.abc import Iterator
from contextlib import contextmanager

from private_ai_stack.config.settings import Settings

_telemetry_configured = False


def configure_telemetry(settings: Settings) -> None:
    global _telemetry_configured
    if _telemetry_configured:
        return
    if not settings.otel_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": settings.otel_service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint)))
        trace.set_tracer_provider(provider)
        _telemetry_configured = True
    except Exception:
        return


@contextmanager
def span(name: str) -> Iterator[None]:
    try:
        from opentelemetry import trace
    except Exception:
        yield
        return

    with trace.get_tracer("private_ai_stack").start_as_current_span(name):
        yield
