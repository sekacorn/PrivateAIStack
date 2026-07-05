# Phase 1 Design Baseline

Status: locked.

## Locked Decisions

- GitHub remote: https://github.com/sekacorn/PrivateAIStack.git
- Default laptop model: `qwen2.5:3b`
- Default runtime profile: `laptop-cpu`
- Default Forge worker count: `1`
- Default startup includes: API, Ollama, PostgreSQL/pgvector
- Optional observability profile: `observability`
- Observability profile includes: OpenTelemetry Collector and Jaeger
- Default command: `docker compose up --build -d`
- Observability command: `docker compose --profile observability up --build -d`
- The application must run correctly when observability is disabled or unavailable.
- Larger models are documented as optional and are never automatically downloaded.
- Public-sector evaluation must consider human ownership of high-impact decisions, accessibility, auditability, records and retention, operational ownership, continuity of operations, vendor portability, measurable user outcomes, and deterministic non-AI alternatives.

## Implementation Milestones

1. Architecture and documentation baseline.
2. Minimal Compose stack with API, PostgreSQL/pgvector, Ollama, health checks, and one Forge task.
3. Persistent memory and local document ingestion/search.
4. Safe-static code-review collector, deterministic tools, normalized findings, and reports.
5. Forge policies, approval handling, audit redaction, and audit export.
6. Optional OpenTelemetry Collector and Jaeger instrumentation.
7. Tests, CI, security scans, dependency-license inventory, examples, and release checklist.

## Acceptance-Test Matrix

| Area | Acceptance test |
| --- | --- |
| Stack startup | `docker compose up --build -d` starts API, Ollama, and PostgreSQL |
| Health | `make health` verifies actual readiness |
| Model | `make model-pull` pulls `qwen2.5:3b` only by explicit command |
| API task | `POST /v1/tasks` creates a Forge-backed task |
| Memory | Ingested document survives restart |
| RAG | Search returns local filename and chunk citation |
| Review | `safe-static` reviews sample repository without modifying it |
| Tools | Ruff or equivalent findings appear in final report |
| Security | Security-tool findings appear in final report |
| Missing tools | Unavailable tools are reported as `not_run` |
| Policy | Denied secret-file read creates an audit event |
| Approval | Approval-required action denies safely without approver |
| Observability | Jaeger trace is available only with `observability` profile |
| Export | Memory and audit data export as JSONL |
| Tests | Test suite requires no paid provider |
| CI | CI covers supported Python versions |
| Docs | Commands are executed or marked unverified |
| Secrets | No real credentials exist in the repository |

## Phase 2 Entry Criteria

- Phase 1 docs exist.
- Forge 0.5.0 integration assumptions are documented.
- Runtime defaults are locked.
- Trust boundaries are documented.
- No unresolved decision blocks minimal stack implementation.

## Verified Implementation Adjustment

Forge 0.5.0 exposes the expected orchestration APIs, but runtime task execution requires awaiting `Orchestrator.run` when it returns an awaitable. PrivateAIStack also registers `qwen2.5:3b` in Forge's `ModelRegistry` and uses fixed routing to the Ollama provider so local tasks do not silently route to hosted providers.
