# Packaging Verification

Date: 2026-07-05

## Scope

Prepare PrivateAIStack for an alpha Python package without publishing to PyPI, pushing container images, deleting Docker volumes, overwriting `.env`, or fabricating verification results.

## Package

- Distribution: `privateaistack`
- Version: `0.1.0a1`
- CLI: `privateaistack`
- Default model: `qwen2.5:3b`
- Default runtime profile: `laptop-cpu`
- Optional observability profile: `observability`

## Manual Publishing Boundary

Publishing remains manual. Use `docs/pypi-release.md` for TestPyPI and PyPI upload commands after local quality checks and artifact inspection pass.

## Verification Log

## Results

- PyPI name check: `https://pypi.org/pypi/privateaistack/json` returned 404 during verification, so `privateaistack` appeared available at that moment. The name is not reserved until upload succeeds.
- `python -m pip install -e ".[dev,security,quality,postgres,observability]"`: passed.
- `python -m ruff check .`: passed.
- `python -m mypy private_ai_stack`: passed.
- `python -m pytest`: passed, 37 tests, 80.96% coverage.
- `python -m bandit -r private_ai_stack`: passed, no issues identified.
- `docker compose config`: passed.
- `docker compose --profile observability config`: passed.
- `python -m build`: passed.
- `python -m twine check dist/*`: passed.
- Wheel install smoke: passed in `.package-test-venv`.
- Wheel CLI smoke: `privateaistack --version`, `privateaistack init package-test-deployment`, `privateaistack config check --directory package-test-deployment`, and `docker compose -f package-test-deployment/compose.yaml config` passed.
- Source distribution install smoke: passed in `.package-sdist-venv`.
- `make quality`: passed.
- `make security`: passed. `pip-audit` skipped `privateaistack` itself because it is not published on PyPI yet; audited dependencies reported no known vulnerabilities.

## Artifacts

- `dist/privateaistack-0.1.0a1-py3-none-any.whl`
  - SHA256: `ED13A3A457211A82C23BFFA00EBD0E8E26FA5F4CD280B43B994D53ADB8E85F68`
- `dist/privateaistack-0.1.0a1.tar.gz`
  - SHA256: `CDEEB6B4ABB3D53F21FAE97341E9E3C4CDFA5E1CDCDA15B3E9EAD4FD16BC8984`

## Artifact Content Check

Wheel and source distribution include the packaged deployment template, including safe `.env.example`, `compose.yaml`, API `Dockerfile`, OpenTelemetry Collector config, PostgreSQL initialization SQL, helper scripts, and placeholder `sample-target/README.md`.

Artifact checks found no real `.env`, virtual environments, audit JSONL logs, backups, exports, generated report payloads, or sample target code.

## Known Limitation

The generated deployment Dockerfile installs `privateaistack[postgres,observability]==0.1.0a1`. That path becomes directly usable after TestPyPI/PyPI publication. Before publishing, use the repository checkout workflow or override the Docker build argument `PRIVATEAISTACK_SPEC`.

## Recommendation

Package ready for TestPyPI.
