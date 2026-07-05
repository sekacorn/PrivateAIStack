# PrivateAIStack

### A one-command private AI stack for local agents, persistent RAG, governed code review, auditability, and open observability.

PrivateAIStack is a local-first v0.1 foundation for running governed AI-agent workflows on infrastructure you control. It uses FastAPI, Forge from `agentforge-oss`, Ollama, PostgreSQL/pgvector, portable audit records, deterministic code-quality tools, and optional OpenTelemetry tracing.

Status: demonstration-grade v0.1 implementation. Production hardening, security review, and organization-specific policy review are required before operational use.

## What It Does

- Runs a FastAPI API with Swagger at `http://127.0.0.1:8000/docs`.
- Uses Ollama with `qwen2.5:3b` by default.
- Persists document chunks in PostgreSQL.
- Reviews local repositories in `safe-static` mode.
- Produces Markdown, JSON, and SARIF-style review reports.
- Records portable JSONL audit events.
- Optionally sends traces through the OpenTelemetry Collector to Jaeger.
- Supports public-sector evaluation needs such as local operation, auditability, human oversight, accessibility, portability, and measurable mission outcomes.

## What It Does Not Do

- It is not a foundation model.
- It is not VM-grade isolation.
- It does not claim regulatory compliance.
- It does not prove reviewed code is secure.
- It does not automatically download large models.
- It does not use hosted providers by default.

## Architecture

```text
User or CI -> FastAPI -> Forge Orchestrator
                        |-> Ollama
                        |-> PostgreSQL/pgvector
                        |-> policy and audit
                        |-> deterministic review tools
                        |-> optional OTLP Collector -> Jaeger
```

## Five-Minute Quickstart

```bash
git clone https://github.com/sekacorn/PrivateAIStack.git
cd PrivateAIStack
cp .env.example .env
docker compose up --build -d
make model-pull
make health
```

Model download is explicit. The default laptop model is `qwen2.5:3b`, selected for a 16 GB laptop profile with about 6 GB free RAM.
On Windows, the Makefile targets call PowerShell scripts in `scripts/`.

Service URLs:

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Bundled Ollama: internal Compose endpoint `http://ollama:11434`
- Host Ollama: `http://127.0.0.1:11434` only if you run Ollama outside Compose
- Jaeger with observability profile: `http://127.0.0.1:16686`

## Optional Observability

```bash
docker compose --profile observability up --build -d
```

Core operation does not require observability. The application should keep working if the collector or Jaeger is unavailable.

## Smoke Checks

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
curl -sS http://127.0.0.1:8000/version
```

## First Task

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/tasks \
  -H "content-type: application/json" \
  -d '{"goal":"Create a safe implementation plan for a local RAG feature."}'
```

## Document Ingestion

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/knowledge/documents \
  -H "content-type: application/json" \
  -d '{"source_name":"mission.md","content":"PrivateAIStack stores local documents and audit records."}'
```

## Knowledge Search

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/knowledge/search \
  -H "content-type: application/json" \
  -d '{"query":"audit records","limit":3}'
```

## Code Review

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/reviews \
  -H "content-type: application/json" \
  -d '{"repository_path":"/app/sample-target","mode":"safe-static"}'
```

Then fetch a report:

```bash
curl -sS "http://127.0.0.1:8000/v1/reviews/REVIEW_ID/report?format=markdown"
```

Review reports focus on Technical Quality and Risk. Public-sector teams should still evaluate whether any proposed AI workflow has a real user need, measurable mission outcome, accessible user path, clear records and retention plan, human ownership for high-impact decisions, operational owner, continuity plan, and a deterministic non-AI alternative where that is safer or simpler.

## Policy Example

Policies deny sensitive file reads, require approval for network access, and log database mutations. Missing approvers deny by default.

```bash
curl -sS http://127.0.0.1:8000/v1/policies
```

The v0.1 HTTP API exposes the policy catalog. Executable policy-gate checks are currently covered through the internal policy engine and tests, not a public policy-test endpoint.

## Backup, Restore, And Export

```bash
make backup
make restore RESTORE_FILE=backups/file.sql
make export-memory
make export-audit
```

Audit, report, and export output is bind-mounted into `audit/`, `reports/`, and `exports/` for local inspection. PostgreSQL and Ollama model storage use Docker named volumes. `docker compose down` preserves named volumes. `docker compose down -v` deletes PostgreSQL and Ollama model data.

## Audit Inspection

```bash
Get-Content audit/audit.jsonl -Tail 20
make export-audit
```

Audit records are JSONL with chained hashes and redacted sensitive fields. They are intended for local review and export, not as a substitute for organization-specific logging controls.

## Shutdown

```bash
docker compose down
```

Do not use `docker compose down -v` unless you intentionally want to delete PostgreSQL and Ollama named-volume data.

## Model Switching

Edit `.env`:

```text
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL=http://ollama:11434
```

For external Ollama, point `OLLAMA_BASE_URL` to your private endpoint. Hosted providers remain disabled unless explicitly configured in future adapters.

## Current Limitations

- The deterministic embedding fallback is private and testable, but not as semantically strong as a dedicated local embedding model.
- Safe-static review does not execute project tests or install dependencies.
- Tool availability depends on the local/container environment.
- Local model output can be wrong and requires human review.
- Application policy is not a replacement for VM or microVM isolation.

## Roadmap

- Stronger local embedding adapters.
- Richer Forge event correlation.
- Hardened sandboxed-execution mode.
- More SARIF normalizers.
- Optional hosted-provider adapters with explicit opt-in.

## License

Original repository code is MIT licensed. Third-party dependencies have their own licenses; see `docs/dependency-licenses.md`.
