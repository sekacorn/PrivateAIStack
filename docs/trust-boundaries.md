# Trust Boundaries

Status: `0.2.0b1` Beta candidate trust-boundary documentation.

```text
User or CI
    |
    | HTTP on localhost by default
    v
PrivateAIStack API container
    |
    +--> read-only reviewed repository mount
    |
    +--> PostgreSQL internal Compose network
    |
    +--> Ollama local/internal endpoint
    |
    +--> OpenTelemetry Collector, optional profile
              |
              v
            Jaeger, optional profile
```

## Boundary Rules

- API is bound to localhost by default.
- PostgreSQL is internal only by default.
- Ollama is local/internal by default.
- Observability is optional and must fail open for runtime availability, not for security policy.
- Reviewed repositories are mounted read-only under a fixed workspace path.
- The system must not mount the Docker socket.
- The system must not require access to the user's home directory.
- Hosted providers are disabled by default.

## Sensitive Paths Excluded From Review

- `.git`
- `.env`
- secret files
- private keys
- cloud credential files
- virtual environments
- `node_modules`
- binary assets
- build outputs
- caches
- coverage outputs
- generated vendor directories
- files larger than the configured threshold

## Approval Boundary

Operations requiring approval:

- External network access.
- Dependency installation.
- Test execution in sandboxed mode.
- Destructive file operations.
- Deleting files.
- Model-provider changes.

If an approver is unavailable, the operation is denied by default.
