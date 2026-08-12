FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG PRIVATEAISTACK_SPEC="privateaistack[postgres,observability]==0.2.0b1"

RUN addgroup --system app && adduser --system --ingroup app app
WORKDIR /app
RUN python -m pip install --upgrade pip && python -m pip install "${PRIVATEAISTACK_SPEC}"

COPY --chown=app:app .env.example ./
USER app
EXPOSE 8000
CMD ["uvicorn", "private_ai_stack.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
