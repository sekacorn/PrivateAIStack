# Private Code Review Design

Status: Beta candidate design and runtime boundary.

The flagship workflow is reviewing a local source-code repository without uploading it to an external AI service.

## Core Rule

Deterministic tools find facts. Agents interpret, correlate, and prioritize those facts.

## Agents

- `ReviewSupervisor`: determines repository language and structure, creates a review plan, delegates bounded tasks, merges findings, removes duplicates, and ranks by severity and confidence.
- `CodeQualityAgent`: reviews maintainability, readability, complexity, error handling, typing, duplication, and architecture.
- `SecurityReviewAgent`: interprets SAST, secret-scan, dependency, and container findings; separates confirmed findings from hypotheses.
- `TestReviewAgent`: interprets tests and coverage, identifies untested critical paths, and reports skipped, flaky, or failing tests honestly.
- `InfrastructureReviewAgent`: reviews Dockerfiles, Compose files, shell scripts, and GitHub Actions.
- `DocumentationReviewAgent`: compares README commands, environment variables, ports, and examples against actual repository files.

## Finding Model

Required fields:

- `id`
- `source_tool`
- `agent`
- `category`
- `severity`
- `confidence`
- `title`
- `explanation`
- `file`
- `start_line`
- `end_line`
- `evidence`
- `remediation`
- `fingerprint`
- `status`

Allowed severity values:

- `critical`
- `high`
- `medium`
- `low`
- `informational`

Allowed confidence values:

- `confirmed`
- `high`
- `medium`
- `speculative`

## Report Sections

The final report must distinguish:

- Technical quality and risk.
- Deterministic confirmed findings.
- Agent-supported interpretations.
- Speculative architectural concerns.
- Tools that did not run.

Product and operational review for public-sector use should be handled as neutral requirements outside the runtime agent set. Teams should confirm user need, measurable mission outcomes, accessibility, human ownership for high-impact decisions, records and retention handling, privacy/security review points, operational ownership, continuity of operations, procurement constraints, vendor portability, and whether deterministic software is more appropriate than AI.

## Safe-Static Defaults

- Read-only repository access.
- No project dependency installation.
- No arbitrary project scripts.
- No untrusted binaries.
- No outbound network calls.
- No hosted provider use without explicit opt-in.
- Sample source remains unchanged.
