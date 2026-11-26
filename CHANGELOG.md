# Changelog

## Unreleased

### Added
- **`agent-memory.v1` export of the knowledge store.**
  `private_ai_stack.memory.portable.export_agent_memory` and
  `privateaistack memory export OUT_DIR` write the store in the versioned format
  the AI Infrastructure Suite reads and scores for the `memory` vector. Each
  chunk becomes one item: `item_id` is the chunk ID, `content_sha256` covers its
  exact text, and the document it came from (`document_id`, `source_name`,
  `document_sha256`) is kept in metadata, with the operator's own metadata under
  `metadata`. Items are written as digested JSONL shards.
- The vectors go into a linked `embedding-portability.v1` document. They use the
  scheme `privateaistack.hashed-bow.sha256-signed`, revision `1`, at the
  configured dimension. Each vector is bound to its item by `content_sha256`,
  and the exact text is retained.
- **Before anything is written**, every stored vector is checked against the
  store's own embedding of its chunk. An inconsistent index (another dimension,
  or a changed value) stops the export and names the chunks.
  `--no-embeddings` exports the text alone.
- **A credential is never exported.** A chunk or metadata value that looks like
  one stops the export. The error names the chunk and the kind, never the value.
  There are two ways to proceed:
  - `--redact-secrets` replaces each detected credential with
    `[REDACTED:<kind>]`, records `redaction_profile: "secrets"`, and computes
    the vectors over the redacted text.
  - `--exclude CHUNK --exclude-reason "..."` leaves the chunk out. Each
    exclusion is written to the audit log as `memory.export_exclusion` with the
    export's digest. An exclusion without a reason or without an audit log is
    refused.

  The detector is AgentForge's, so both writers refuse the same things.
- The export goes only into an empty, non-linked directory, and files are
  created exclusively.
- No tombstones are written: this store deletes replaced chunks without leaving
  a marker, so every export reports `deleted_by_policy: 0`.
- `MemoryStore.all_chunks()`: a public read of every chunk and its stored
  vector.

### Changed
- **The `agentforge-oss` floor is now 0.5.9** (`>=0.5.9,<0.6.0`, in the core
  requirement and in every extra). The export reuses
  `forge.security.secrets`, which first shipped in 0.5.9.

### Deprecated
- `MemoryStore.export_jsonl` is superseded by `export_agent_memory`. It still
  works unchanged, but now emits a `DeprecationWarning`. Its output has no
  version field, no schema and no digests.

## 0.2.0b4 - 2026-11-25

A bug-fix release. Knowledge search over vectors of a different dimension now
fails with an explicit `409` instead of silently ranking those chunks last.

### Fixed
- **Knowledge search no longer hides vectors of a different dimension.**
  `cosine_similarity` returned `0.0` when the query vector and a stored
  vector had different lengths, so a store built at another embedding
  dimension (or holding two) ranked every such chunk last **without any
  error**. **Affects every published version, 0.1.0a1 through 0.2.0b3.** It
  now raises `EmbeddingDimensionError` (a `ValueError`), and
  `POST /v1/knowledge/search` answers `409 knowledge_index_inconsistent`
  instead of returning wrong results. The response names no stored text.

## 0.2.0b3 - 2026-11-24

A security release. **If your `OLLAMA_BASE_URL` or OTel endpoint ever
contained credentials or a query-string token, rotate them**: such a URL is
now refused. Nothing else about the runtime changes.

### Security

- **`ollama_base_url` and `otel_endpoint` refuse credentials, query strings
  and fragments. Affects every published version through 0.2.0b2.** A URL such
  as `http://user:password@ollama:11434` or `http://ollama:11434?key=...` was
  accepted and used. The HTTP client logs each request URL at INFO, and the
  Ollama URL is passed on to Forge, so a secret placed there reached any log
  that captured INFO output. Such a URL is now refused when the settings
  load, with a message that never repeats it. Rejected settings are no longer
  echoed in validation errors at all, which also covers a malformed
  `database_url`. If you ran an earlier version with a secret in either URL
  and INFO logging on, rotate it. No setting needs a query string, so none is
  allowed.

## 0.2.0b2 - 2026-11-24

### Re-embedding

New in this release: `private_ai_stack.reembed` and the `privateaistack
reembed` command group. No earlier published version contains either, so every
item here is new to anyone upgrading from `0.2.0b1`.

**Reading the records it writes requires ai-infrastructure-suite 0.1.0b3 or
later.** This is a compatibility statement, not a dependency: this package does
not depend on the suite.

- **`privateaistack reembed run`** rebuilds an `embedding-portability.v1`
  store with a different model, from the store's retained source text. It
  writes the rebuilt store as an ordinary `embedding-portability.v1` document,
  so `ai-suite exit-report` scores it with no new code, and a
  `re-embed-record.v1` describing what was rebuilt, what was skipped, how it
  was batched and what left the machine.
- **Resumable batches.** A batch's shard reaches disk before the checkpoint
  claims it, so a crash can leave an unclaimed shard but never a claim without
  its shard. A changed claimed shard is an error, and a checkpoint written for
  a different input, target, model or batch size is refused. Tests interrupt a
  run at every point it can be interrupted and compare the result byte for
  byte with an uninterrupted run. One batch is held in memory at a time;
  the default is 64 records.
- **`availability` has no default.** It is an operator attestation that no
  offline check can confirm.
- **`--estimate-only`** embeds a small calibration sample and extrapolates to
  a `re-embed-estimate.v1` document, labelling which parts are exact,
  measured and extrapolated. It writes no shard, checkpoint or store. Sample
  records are chosen by a recorded seed rather than by identifier.
- **Cost in money comes from AIMeter or not at all**, through the new `[cost]`
  extra (`aimeter-oss>=0.2.0b1,<0.3.0`, the same range as the suite).
- **`privateaistack reembed quality`** compares retrieval before and after a
  rebuild and adds a `quality` block to the record. It measures neighborhood
  agreement, not search quality. Thresholds are declared by the operator and
  always shown; with no agreement floor declared the status is `unjudged`, a
  result with metrics and no verdict. `reembed run` writes no quality block
  at all.
- **An egress gate, default-deny for anything but built-in code.** An injected
  executor needs `allow_external`; hosted execution also needs
  `allow_hosted`, an HTTPS endpoint and a `public` or `internal` data
  classification, and is refused for `unknown`, `confidential` and
  `restricted` data. The refusal happens before the evidence document is
  opened. Every run records an egress block, including what was submitted to
  an external executor; submission is recorded, delivery stays `unverified`.
  No hosted provider ships in this package.

- **`privateaistack reembed quality --anchor-seed` marks the anchor set
  operator-chosen and never moves a lock-in score.** The seed is derived as
  `sha256` over the source and target store digests; supplying `--anchor-seed`
  overrides that derivation. An overridden seed sets `anchor_selection:
  operator`: the comparison still runs, and its label, metrics and threshold
  checks are all reported in full, but the result can never move a store from
  50 to 25, and no seed is written into the re-embed record.

  The anchors drawn are identical either way. Only the provenance differs, and
  the provenance is the point: an operator who can choose the seed can draw
  query set after query set until the numbers clear the declared threshold and
  then publish the one that passed. A derived seed is a function of content the
  record already commits to, so nobody can steer it and a third party can
  recompute the same set from the record alone.

  Omit the flag to get a derived seed and a `seeded_random` set. The AI
  Infrastructure Suite derives the same value independently and reads any other
  seed as operator-chosen, so the two must agree; a cross-repo test asserts
  that they do.

### Release process

- The release workflow gained a `dry_run` input (default `true`) on
  `workflow_dispatch`, a verify job that runs lint, type check, Bandit and
  tests before the build, and a `v[0-9]*` tag trigger. The publish job now
  waits for a required reviewer at the `pypi` environment.

### Dependencies

- **Raised the minimum Forge version to `agentforge-oss>=0.5.5,<0.6.0`**, for
  the core install and every extra that names it. 0.5.5 is a security release:
  in 0.5.4 and earlier, provider errors could repeat upstream text, including
  an API key reflected by a proxy or third-party endpoint, and a `base_url`
  containing credentials was logged. The previous floor, `0.5.1`, was also
  lower than anything CI verified: the `embedder-conformance` job always
  tests the latest release in the range.

## 0.2.0b1 - 2026-08-12

### First Beta candidate

- Made PostgreSQL-backed knowledge persistence explicit at startup and retained `memory://local` only as an explicit volatile test/development mode.
- Disabled direct Ollama fallback by default; it now requires an explicit degraded-mode configuration because it is not Forge-equivalent execution.
- Hardened local audit append and verification behavior for cooperating writers, malformed records, and bounded records.
- Staged safe-static review inputs, bounded tool output and total review time, and clarified that static tools are local subprocesses rather than a VM sandbox.
- Tightened configuration, API-key comparison, request/input limits, deterministic retrieval ordering, and public operational-boundary documentation.

The Beta candidate retains documented operational limits: Compose is development-oriented, tasks and reviews are process-local, audit chains are tamper-evident rather than immutable, and backup/restore and telemetry have not yet been live-verified.

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
