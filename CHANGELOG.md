# Changelog

## Unreleased

### Beta-hardening preparation

- Made PostgreSQL-backed knowledge persistence explicit at startup and retained `memory://local` only as an explicit volatile test/development mode.
- Disabled direct Ollama fallback by default; it now requires an explicit degraded-mode configuration because it is not Forge-equivalent execution.
- Hardened local audit append and verification behavior for cooperating writers, malformed records, and bounded records.
- Staged safe-static review inputs, bounded tool output and total review time, and clarified that static tools are local subprocesses rather than a VM sandbox.
- Tightened configuration, API-key comparison, request/input limits, deterministic retrieval ordering, and public operational-boundary documentation.

## 0.1.0a3 - 2026-08-06

- Publish corrected Apache-2.0 package metadata under the public maintainer identity `sekacorn`.
- Preserve the existing local-first runtime and documented deployment templates without claiming new integrations.

## 0.1.0a2

### Release infrastructure

- Reconnected release metadata and workflows for the recreated `sekacorn/PrivateAIStack` repository and the existing `privateaistack` PyPI project.
- Restored package publication through a tag-triggered GitHub Actions workflow designed for PyPI Trusted Publishing.
- Updated package metadata and project URLs for the current repository.

### Dependencies

- Raised the minimum supported Forge version to `agentforge-oss>=0.5.1,<0.6.0` so packaged pgvector functionality is available.

### Packaging

- Updated the package version, deployment template metadata, and generated deployment Dockerfile default package spec to `0.1.0a2`.

### Documentation

- Updated installation instructions to reflect the published PyPI distribution name, Python import package, and CLI command.

## 0.1.0a1

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
