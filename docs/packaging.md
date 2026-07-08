# Packaging

PrivateAIStack publishes as the Python distribution `privateaistack`.

Current alpha version: `0.1.0a2`.

The package contains:

- Python modules under `private_ai_stack/`.
- The `privateaistack` CLI entry point.
- Type marker `private_ai_stack/py.typed`.
- Deployment templates under `private_ai_stack/templates/deployment/`.

The package must not contain local secrets, `.env`, models, Docker volumes, virtual environments, generated cache directories, backups, exports, or generated report output.

## Build

```bash
python -m pip install -e ".[dev,security,quality]"
python -m build
python -m twine check dist/*
```

## Inspect Contents

```bash
python -m zipfile -l dist/privateaistack-0.1.0a2-py3-none-any.whl
tar -tf dist/privateaistack-0.1.0a2.tar.gz
```

Check that template files are present and local runtime artifacts are absent.

## Fresh Install Smoke

```bash
python -m venv .package-test-venv
.package-test-venv/Scripts/python -m pip install --upgrade pip
.package-test-venv/Scripts/python -m pip install dist/privateaistack-0.1.0a2-py3-none-any.whl
.package-test-venv/Scripts/privateaistack --version
.package-test-venv/Scripts/privateaistack init package-test-deployment
.package-test-venv/Scripts/privateaistack config check --directory package-test-deployment
docker compose -f package-test-deployment/compose.yaml config
```

Do not start the generated stack during packaging verification if another local runtime is already running.
