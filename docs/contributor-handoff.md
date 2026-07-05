# Contributor Handoff

Start here:

1. Read `README.md`, `ARCHITECTURE.md`, and `THREAT_MODEL.md`.
2. Install development dependencies with `python -m pip install -e ".[dev,security,quality,postgres,observability]"`.
3. Run `make test`, `make lint`, and `make typecheck`.
4. Start the stack with `docker compose up --build -d`.
5. Pull the model with `make model-pull`.
6. Try the sample review against `sample-target`.
7. Before release work, read `docs/packaging.md`, `docs/cli.md`, `docs/template-versioning.md`, and `docs/pypi-release.md`.

Good first issues:

- Add richer SARIF normalizers.
- Add a stronger local embedding adapter.
- Expand policy examples.
- Improve Forge event correlation.
- Add integration tests for PostgreSQL/pgvector in CI.
- Expand CLI integration tests against an isolated generated deployment.
