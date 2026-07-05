from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette import status


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    trace_id: str | None = None
    details: Mapping[str, Any] = Field(default_factory=dict)


class AppError(Exception):
    def __init__(self, error: str, message: str, status_code: int = 400, details: Mapping[str, Any] | None = None) -> None:
        self.error = error
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.error,
                message=exc.message,
                request_id=_request_id(request),
                trace_id=getattr(request.state, "trace_id", None),
                details=exc.details,
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                message="Request validation failed.",
                request_id=_request_id(request),
                trace_id=getattr(request.state, "trace_id", None),
                details={"errors": exc.errors()},
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                message="An internal error occurred. See operational logs for details.",
                request_id=_request_id(request),
                trace_id=getattr(request.state, "trace_id", None),
                details={"type": exc.__class__.__name__},
            ).model_dump(),
        )
