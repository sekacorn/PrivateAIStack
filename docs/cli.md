# CLI

The CLI command is `privateaistack`.

Exit codes:

- `0`: success.
- `1`: runtime failure.
- `2`: invalid input or invalid configuration.
- `3`: required dependency unavailable.
- `4`: partial or degraded state.

Commands:

```bash
privateaistack --version
privateaistack doctor --directory .
privateaistack init ./deployment
privateaistack config check --directory .
privateaistack up --directory .
privateaistack up --directory . --observability
privateaistack down --directory .
privateaistack status --directory .
privateaistack logs --directory .
privateaistack health --directory .
privateaistack model list --directory .
privateaistack model pull --directory . --model qwen2.5:3b
privateaistack task run --directory . "Summarize the operating plan."
privateaistack knowledge add --directory . --content "Local document text"
privateaistack knowledge search --directory . "local document"
privateaistack review --directory . /app/sample-target
privateaistack audit show --directory .
privateaistack audit export --directory .
privateaistack audit verify --directory .
privateaistack backup --directory .
privateaistack restore --directory . backups/private_ai_stack.sql
```

The model commands operate inside the Docker Compose Ollama service by default. A model installed in a host Ollama service is not automatically present in the Compose named volume.

Commands use subprocess argument arrays and do not interpolate shell strings.
