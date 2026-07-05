#!/usr/bin/env sh
set -eu
mkdir -p backups
docker compose exec -T postgres pg_dump -U private_ai_stack private_ai_stack > "backups/private_ai_stack_$(date +%Y%m%d%H%M%S).sql"
