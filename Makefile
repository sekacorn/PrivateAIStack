.PHONY: bootstrap model-pull up up-observability down logs health smoke test lint typecheck security quality build-package code-review code-review-demo backup restore export-memory export-audit clean

bootstrap:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bootstrap.ps1

model-pull:
	docker compose exec ollama ollama pull $${OLLAMA_MODEL:-qwen2.5:3b}

up:
	docker compose up --build -d

up-observability:
	docker compose --profile observability up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

health:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/health.ps1

smoke:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/smoke.ps1

test:
	python -m pytest

lint:
	python -m ruff check .

typecheck:
	python -m mypy private_ai_stack

security:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/security.ps1

quality: lint typecheck test security

build-package:
	python -m build
	python -m twine check dist/*

code-review:
	curl -sS -X POST http://127.0.0.1:8000/v1/reviews -H "content-type: application/json" -d '{"repository_path":"/workspace/target","mode":"safe-static"}'

code-review-demo:
	curl -sS -X POST http://127.0.0.1:8000/v1/reviews -H "content-type: application/json" -d '{"repository_path":"/app/sample-target","mode":"safe-static"}'

backup:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/backup.ps1

restore:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/restore.ps1

export-memory:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/export.ps1 memory

export-audit:
	powershell -NoProfile -ExecutionPolicy Bypass -File scripts/export.ps1 audit

clean:
	powershell -NoProfile -Command "Remove-Item -Recurse -Force .pytest_cache,.mypy_cache,.ruff_cache,htmlcov -ErrorAction SilentlyContinue; Remove-Item -Force coverage.xml -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force reports | Out-Null; New-Item -ItemType File -Force reports/.gitkeep | Out-Null"
