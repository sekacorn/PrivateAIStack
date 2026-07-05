# Dependency Licenses

PrivateAIStack original code is MIT licensed. Dependencies retain their own licenses.

Initial inventory:

| Dependency | Purpose | License notes |
| --- | --- | --- |
| agentforge-oss | Forge orchestration and governance | MIT according to PyPI metadata inspected during Phase 1 |
| FastAPI | API framework | Check installed package metadata before release |
| Pydantic | Validation | Check installed package metadata before release |
| Uvicorn | ASGI server | Check installed package metadata before release |
| httpx | HTTP client | Check installed package metadata before release |
| psycopg | PostgreSQL client | Check installed package metadata before release |
| OpenTelemetry packages | Tracing | Check installed package metadata before release |
| Ruff, mypy, pytest, Bandit, pip-audit | Development and quality tooling | Development dependencies; check metadata before release |
| Ollama container | Local model server | Distributed separately through container image |
| pgvector container | PostgreSQL + vector extension | Distributed separately through container image |
| OpenTelemetry Collector container | Optional observability | Distributed separately through container image |
| Jaeger container | Optional tracing UI | Distributed separately through container image |

Before tagged releases, regenerate this inventory from installed package metadata and container image notices.
