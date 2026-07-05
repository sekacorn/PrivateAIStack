#!/usr/bin/env sh
set -eu

for cmd in docker curl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
else
  echo ".env exists; not overwriting"
fi

MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
echo "Configured model: $MODEL"
echo "Model download is explicit. Run 'make model-pull' when you are ready."

docker compose up --build -d
./scripts/health.sh
./scripts/smoke.sh || true

echo "API: http://127.0.0.1:8000"
echo "Swagger: http://127.0.0.1:8000/docs"
echo "Jaeger, when observability is enabled: http://127.0.0.1:16686"
