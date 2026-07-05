FROM python:3.13-slim-bookworm AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY pyproject.toml README.md ./
COPY private_ai_stack ./private_ai_stack
RUN python -m pip install --upgrade pip && python -m pip wheel --wheel-dir /wheels ".[postgres,observability]"

FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links /wheels "privateaistack[postgres,observability]" && rm -rf /wheels
COPY --chown=app:app .env.example ./
USER app
EXPOSE 8000
CMD ["uvicorn", "private_ai_stack.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
