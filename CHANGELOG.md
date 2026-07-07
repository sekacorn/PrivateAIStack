# Changelog

## Unreleased

- Prepared Python package metadata for the `privateaistack` alpha distribution at version `0.1.0a1`.
- Added the `privateaistack` CLI for deployment initialization, Compose management, health checks, model management, task/review/knowledge API calls, audit inspection, backup, and restore.
- Added packaged deployment templates with optional `observability` profile support and `.env` preservation.
- Added packaging, CLI, PyPI release, and template-versioning documentation plus packaging CI validation.
- Updated the pinned official Ollama Docker image from `ollama/ollama:0.5.7` to `ollama/ollama:0.31.1`, the newest verified stable non-release-candidate tag available from Docker Hub at verification time.
- Removed the mistakenly shipped Government Business Analyst runtime agent while preserving neutral public-sector product requirements in documentation.
- Repository maintenance: consolidated git commit authorship under a single maintainer identity (`sekacorn`).

## 0.1.0

- Initial local-first PrivateAIStack v0.1 implementation.
- FastAPI API, Ollama integration, PostgreSQL-backed document storage, audit JSONL, safe-static code review, optional observability, and open-source project scaffolding.
