$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path backups | Out-Null
$stamp = Get-Date -Format "yyyyMMddHHmmss"
docker compose exec -T postgres pg_dump -U private_ai_stack private_ai_stack | Set-Content -Encoding UTF8 "backups/private_ai_stack_$stamp.sql"
