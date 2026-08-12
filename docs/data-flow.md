# Data Flow

Status: `0.2.0b1` Beta candidate data-flow documentation.

## Agent Task Flow

```text
POST /v1/tasks
    |
    v
Validate request
    |
    v
Create task ID, request ID, trace context
    |
    v
Evaluate policy
    |
    v
Forge Orchestrator
    |
    +--> model provider
    +--> tool sandbox
    +--> audit event
    +--> telemetry span, optional
    |
    v
Task result
```

## Document Ingestion Flow

```text
POST /v1/knowledge/documents
    |
    v
Validate path/content
    |
    v
Chunk document
    |
    v
Hash content
    |
    v
Idempotency check
    |
    v
Embed locally
    |
    v
Store in PostgreSQL / pgvector
    |
    v
Audit ingestion event
```

## Code Review Flow

```text
POST /v1/reviews
    |
    v
Validate review mode
    |
    v
Collect repository files safely
    |
    v
Run deterministic tools
    |
    v
Normalize findings
    |
    v
Forge supervisor and specialist agents
    |
    v
Deduplicate and rank findings
    |
    v
Generate Markdown, JSON, SARIF where possible
    |
    v
Hash reports and record audit event
```

## Export Flow

```text
Export request
    |
    v
Authorize and evaluate policy
    |
    v
Read memory or audit records
    |
    v
Redact where required
    |
    v
Write portable JSONL export
    |
    v
Record export audit event
```

## Data That Must Not Leave Local Defaults

- Source code under review.
- Local RAG documents.
- Secrets or credentials.
- Private prompts with source excerpts.
- Database credentials.

Hosted providers can only be used after explicit opt-in configuration.
