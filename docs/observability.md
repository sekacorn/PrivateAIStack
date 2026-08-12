# Observability

Observability is optional in the current Beta candidate and remains disabled by default.

Default startup:

```bash
docker compose up --build -d
```

Observability startup:

```bash
docker compose --profile observability up --build -d
```

The API exports to the OpenTelemetry Collector boundary when enabled. Jaeger is only used behind the collector and is not required for core operation.

Do not place secrets, full source files, private prompts, database passwords, API keys, or credential files in trace attributes.
