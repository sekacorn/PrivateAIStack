#!/usr/bin/env sh
set -eu
if [ "${1:-}" = "" ]; then
  echo "Usage: ./scripts/restore.sh backups/file.sql" >&2
  exit 1
fi
docker compose exec -T postgres psql -U private_ai_stack private_ai_stack < "$1"
