# PrivateAIStack Architecture

PrivateAIStack is a local-first, provider-replaceable AI stack for governed agent tasks, persistent RAG, private code review, auditability, and optional open observability.

Status: Phase 1 design baseline.

Author: sekacorn

Project contact: Open a GitHub repository issue for general project questions.

Security contact: Use GitHub Security Advisories if enabled, or contact the maintainer through the repository without posting sensitive details publicly.

## V0.1 Decisions

- Repository: https://github.com/sekacorn/PrivateAIStack.git
- Original repository code license: MIT
- Default laptop model: `qwen2.5:3b`
- Default runtime profile: `laptop-cpu`
- Default Forge worker count: `1`
- Default services: API, Ollama, PostgreSQL with pgvector
- Optional observability profile: `observability`
- Optional observability services: OpenTelemetry Collector and Jaeger
- Default startup command: `docker compose up --build -d`
- Observability startup command: `docker compose --profile observability up --build -d`
- Larger models are optional and are never automatically downloaded.

## Logical Architecture

```text
User or CI
    |
    v
PrivateAIStack FastAPI service
    |
    v
Application service layer
    |
    +--> Task service
    +--> Knowledge service
    +--> Code-review service
    +--> Audit/export service
    +--> Policy/approval adapter
    |
    v
Forge Orchestrator
    |
    +--> Supervisor agent
    |      |
    |      +--> specialist agents
    |
    +--> Forge policy engine
    +--> Forge tool sandbox
    +--> Forge audit log
    +--> Forge event bus / OpenTelemetry
    +--> Ollama or configured model provider
    +--> PostgreSQL / pgvector
```

## Observability Flow

```text
PrivateAIStack and Forge
    |
    | OTLP, when enabled
    v
OpenTelemetry Collector
    |
    v
Jaeger
```

The API must keep working when observability is disabled or unavailable. Application code exports to the OpenTelemetry Collector boundary rather than hardwiring directly to Jaeger.

## Component Responsibilities

FastAPI owns:

- HTTP contracts and Swagger documentation.
- Request IDs, trace IDs, structured errors, and exception redaction.
- Task and review lifecycle APIs.
- Safe-static repository collection.
- Report, audit, memory, backup, and export APIs.

Forge owns or fronts:

- Agent orchestration.
- Supervisor and worker execution.
- Model routing.
- Tool sandbox policy checks.
- Governance policies.
- Audit primitives.
- Event bus and OpenTelemetry hooks.
- pgvector-backed memory where compatible.

PostgreSQL with pgvector owns:

- Persistent RAG records.
- Agent memory metadata.
- Exportable memory state.
- Database backup and restore target.

Ollama owns:

- Local model serving.
- Explicit model pulls.
- Provider-replaceable inference endpoint.

Deterministic quality tools own:

- Linting, type checking, SAST, dependency scanning, secret scanning, complexity analysis, infrastructure checks, and repository metadata facts.

Agents own:

- Interpretation, correlation, deduplication, prioritization, and report synthesis.

## Forge Integration Baseline

The minimum supported Forge package for v0.1 is expected to be:

```text
agentforge-oss>=0.5.1,<0.6.0
```

Verified import:

```python
import forge
```

Verified local Forge 0.5.1-compatible exports include:

- `Orchestrator`
- `Agent`
- `Supervisor`
- `ForgeConfig`
- `OllamaProvider`
- `ModelRegistry`
- `PGVectorMemoryStore`
- `PolicySet`
- `PolicyRule`
- `ToolSandbox`
- `AuditLogger`
- `EventBus`

Implementation note: Forge `Orchestrator.run` may return an awaitable at runtime. PrivateAIStack awaits that result and explicitly registers the configured Ollama model in a `ModelRegistry` with fixed routing to avoid accidental hosted-provider fallback.

Integration points requiring implementation-time verification:

- Async context manager behavior around `Orchestrator`.
- Exact audit write and verification APIs.
- Event subscription APIs for correlation with FastAPI request and task IDs.
- pgvector embedding dimension validation behavior.
- Whether PrivateAIStack needs a subprocess wrapper around deterministic tools in addition to Forge `ToolSandbox`.

## V0.1 API Surface

Required endpoints:

- `GET /health`
- `GET /ready`
- `GET /version`
- `POST /v1/tasks`
- `GET /v1/tasks/{task_id}`
- `GET /v1/tasks/{task_id}/events`
- `GET /v1/tasks/{task_id}/audit`
- `POST /v1/knowledge/documents`
- `POST /v1/knowledge/search`
- `POST /v1/reviews`
- `GET /v1/reviews/{review_id}`
- `GET /v1/reviews/{review_id}/report`
- `GET /v1/reviews/{review_id}/findings`
- `GET /v1/models`
- `GET /v1/policies`

## Code Review Architecture

```text
Review request
    |
    v
Safe repository collector
    |
    +--> path allowlist and exclusion checks
    +--> file size limits
    +--> read-only access
    |
    v
Deterministic tool runners
    |
    +--> Ruff, mypy, pytest when approved, Bandit, pip-audit, Radon
    +--> Hadolint, ShellCheck, yamllint, markdownlint, actionlint, Trivy
    +--> detect-secrets or Gitleaks
    |
    v
Normalized findings
    |
    v
ReviewSupervisor
    |
    +--> CodeQualityAgent
    +--> SecurityReviewAgent
    +--> TestReviewAgent
    +--> InfrastructureReviewAgent
    +--> DocumentationReviewAgent
    |
    v
Markdown, JSON, SARIF, machine-readable summary
```

Rule: deterministic tools find facts; agents interpret, correlate, and prioritize those facts.

Public-sector and restricted-environment deployments must still be evaluated against product requirements beyond code findings: real user need, measurable mission outcomes, accessible and plain-language interfaces, human ownership for high-impact decisions, records and retention handling, operational ownership, continuity of operations, vendor portability, and whether deterministic software is safer or simpler than AI.

## Runtime Profiles

### laptop-cpu

- Target: 16 GB system RAM with about 6 GB free.
- Default model: `qwen2.5:3b`.
- Workers: `1`.
- Context: limited.
- Observability: optional.

### workstation-gpu

- Target: 32 GB or more system RAM and compatible GPU.
- Larger Ollama model permitted after explicit user pull.
- More workers allowed by configuration.
- Observability recommended.

### external-ollama

- PrivateAIStack services run in Compose.
- Ollama runs on host or another private server.
- `OLLAMA_BASE_URL` selects the external endpoint.

### hosted-provider-opt-in

- Disabled by default.
- Requires explicit environment configuration.
- Documentation must warn that content may leave the local environment.

## Design Constraints

- No custom React or Next.js frontend in v0.1.
- No automatic large model download.
- No database port published by default.
- No Docker socket mount.
- No privileged containers.
- No host root mount.
- No hosted provider unless explicitly opted in.
- No claim of regulatory certification or complete sandboxing.
- No fabricated test results.
