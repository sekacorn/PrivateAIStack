# Backup And Restore

Back up PostgreSQL:

```bash
make backup
```

Restore:

```bash
./scripts/restore.sh backups/file.sql
```

Export memory:

```bash
make export-memory
```

Export audit:

```bash
make export-audit
```

`docker compose down` preserves named volumes. `docker compose down -v` removes PostgreSQL data and Ollama model storage.
