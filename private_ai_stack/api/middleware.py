import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from private_ai_stack.config.settings import get_settings
from private_ai_stack.observability.telemetry import span


async def request_context_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    with span(f"http {request.method} {request.url.path}"):
        configured_limit = get_settings().max_request_id_chars
        candidate = request.headers.get("x-request-id")
        request_id = candidate if candidate and len(candidate) <= configured_limit and candidate.isprintable() else str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.trace_id = request.headers.get("traceparent", "").split("-")[1] if request.headers.get("traceparent") else None
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        if request.state.trace_id:
            response.headers["x-trace-id"] = request.state.trace_id
        return response
