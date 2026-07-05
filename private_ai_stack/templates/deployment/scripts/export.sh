#!/usr/bin/env sh
set -eu
mkdir -p exports
case "${1:-}" in
  memory)
    docker compose exec -T postgres psql -U private_ai_stack -d private_ai_stack -c "COPY (SELECT row_to_json(document_chunks) FROM document_chunks) TO STDOUT" > exports/memory.jsonl
    ;;
  audit)
    docker compose cp api:/app/audit/audit.jsonl exports/audit.jsonl
    ;;
  *)
    echo "Usage: ./scripts/export.sh memory|audit" >&2
    exit 1
    ;;
esac
