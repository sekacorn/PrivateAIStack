# PrivateAIStack

### A local-first AI stack for controlled model execution, explicit RAG persistence, auditability, and reproducible development deployment.

PrivateAIStack is a local-first v0.1 foundation for running governed AI-agent workflows on infrastructure you control. It uses FastAPI, Forge from `agentforge-oss`, Ollama, PostgreSQL/pgvector, portable audit records, deterministic code-quality tools, and optional OpenTelemetry tracing.

Status: alpha package `0.1.0a3`. Production hardening, security review, and organization-specific policy review are required before operational use.

## What It Does

- Runs a FastAPI API with Swagger at `http://127.0.0.1:8000/docs`.
- Uses Ollama with `qwen2.5:3b` by default.
- Uses PostgreSQL for durable document chunks by default; startup fails when that configured store is unavailable.
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
- Local execution is not automatically secure execution.
- Development Compose defaults require operator hardening before production use.

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

Install the published alpha package:

```bash
python -m pip install "privateaistack==0.1.0a3"
privateaistack --version
```

General installation can use the latest published version:

```bash
python -m pip install privateaistack
```

Package identity:

- PyPI distribution: `privateaistack`
- Python import package: `private_ai_stack`
- CLI command: `privateaistack`

For development from a local checkout:

```bash
python -m pip install -e ".[dev,postgres,observability]"
privateaistack --version
```

Create a standalone deployment template:

```bash
privateaistack init ./privateaistack-deploy
cd privateaistack-deploy
privateaistack config check --directory .
privateaistack up --directory .
privateaistack model pull --directory . --model qwen2.5:3b
privateaistack health --directory .
```

The packaged template includes `compose.yaml`, `.env.example`, OpenTelemetry Collector config, PostgreSQL initialization SQL, helper scripts, and local `audit/`, `reports/`, `exports/`, and `backups/` directories. Compose loads `.env.example` and optional `.env` overrides. `privateaistack init` refuses non-empty directories unless `--force` is used and never overwrites an existing `.env` file.

Repository checkout workflow:

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
# or
privateaistack up --directory . --observability
```

Core operation does not require observability. The application should keep working if the collector or Jaeger is unavailable.
OpenTelemetry is disabled by default. The bundled Collector and Jaeger configuration is for local development and is not live-verified by the unit tests.

## Smoke Checks

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
curl -sS http://127.0.0.1:8000/version
privateaistack doctor --directory .
```

## First Task

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/tasks \
  -H "content-type: application/json" \
  -d '{"goal":"Create a safe implementation plan for a local RAG feature."}'
privateaistack task run --directory . "Create a safe implementation plan for a local RAG feature."
```

Task records and task events are process-local. A task is created and executed synchronously by the current API process, and task/review lookup state does not survive an API restart. Audit records and PostgreSQL-backed documents have separate persistence behavior.

Forge is the normal task path and uses fixed local Ollama routing. If Forge fails, PrivateAIStack fails the task by default. Set `ALLOW_DIRECT_OLLAMA_FALLBACK=true` only when an operator deliberately accepts a degraded direct-Ollama path; that path stays local and is audited, but is not Forge-equivalent policy, routing, limit, or telemetry behavior.

## Document Ingestion

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/knowledge/documents \
  -H "content-type: application/json" \
  -d '{"source_name":"mission.md","content":"PrivateAIStack stores local documents and audit records."}'
privateaistack knowledge add --directory . --source-name mission.md --content "PrivateAIStack stores local documents and audit records."
```

`DATABASE_URL=postgresql://...` is the durable default. If PostgreSQL cannot initialize, the API does not start and cannot falsely present volatile data as persistent. `DATABASE_URL=memory://local` is an explicit test/development mode: documents are process-local and `/ready` reports a degraded `persistence: volatile` check.

## Knowledge Search

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/knowledge/search \
  -H "content-type: application/json" \
  -d '{"query":"audit records","limit":3}'
privateaistack knowledge search --directory . "audit records" --limit 3
```

Chunks and hash embeddings are deterministic, and equal retrieval scores are ordered by chunk ID. The default deterministic hash embedding is a lightweight local/test mechanism, not a dedicated semantic embedding model. PostgreSQL stores embedding JSON and the application performs cosine ranking in Python; pgvector is installed for deployment compatibility but this release does not claim database-side ANN search.

## Code Review

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/reviews \
  -H "content-type: application/json" \
  -d '{"repository_path":"/app/sample-target","mode":"safe-static"}'
privateaistack review --directory . /app/sample-target
```

Then fetch a report:

```bash
curl -sS "http://127.0.0.1:8000/v1/reviews/REVIEW_ID/report?format=markdown"
```

`safe-static` copies only selected regular files into a temporary analysis tree before invoking its fixed tool list. Secret paths, symlinks, binary files, and oversized files are excluded from that copy. Tool output and the total review runtime are bounded. The tools still execute as local subprocesses against copied repository content; safe-static is not VM-level sandboxing, does not prove code safety, and does not execute the target project. `sandboxed-execution` remains intentionally unavailable pending an explicit approval and isolation implementation.

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
privateaistack backup --directory .
privateaistack restore --directory . backups/file.sql
privateaistack audit export --directory .
```

Audit, report, and export output is bind-mounted into `audit/`, `reports/`, and `exports/` for local inspection. PostgreSQL and Ollama model storage use Docker named volumes. `docker compose down` preserves named volumes. `docker compose down -v` deletes PostgreSQL and Ollama model data.

## Audit Inspection

```bash
Get-Content audit/audit.jsonl -Tail 20
make export-audit
privateaistack audit show --directory . --lines 20
privateaistack audit verify --directory .
```

Audit records are JSONL with chained hashes and redacted sensitive fields. They are intended for local review and export, not as a substitute for organization-specific logging controls.
Cooperating local writers serialize appends and verify the existing chain before extending it. The chain is tamper-evident rather than immutable evidence, externally anchored provenance, or an AIAuditLog runtime integration.

Backup and restore commands operate on the Compose PostgreSQL service only. They do not back up audit bind mounts, reports, exports, configuration, or Ollama model volumes. The scripts are convenience tooling; live backup/restore verification depends on an operator-run Compose deployment.

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
- API-key authentication is one shared-key authentication mechanism, not authorization, tenancy, or identity management. It is optional in development and required by configuration for `production` environment mode.
- `/health` means the API process is alive. `/ready` checks the configured Ollama endpoint/model and persistence state; `memory://local` is intentionally reported as degraded rather than durable readiness.
- Compose uses development credentials and local bind mounts. Operators must supply secrets, network policy, TLS, retention, backup validation, and production deployment controls.
- Review reports, task state, and review state are process-local unless an external durable system is added.

## Roadmap

- Stronger local embedding adapters.
- Richer Forge event correlation.
- Hardened sandboxed-execution mode.
- More SARIF normalizers.
- Optional hosted-provider adapters with explicit opt-in.

## License

Original repository code is Apache-2.0 licensed. Third-party dependencies have their own licenses; see `docs/dependency-licenses.md`.
