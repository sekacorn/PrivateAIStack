$ErrorActionPreference = "Stop"
if (-not $env:RESTORE_FILE) {
    throw "Set RESTORE_FILE to the SQL backup path before running make restore."
}
Get-Content -Raw -LiteralPath $env:RESTORE_FILE | docker compose exec -T postgres psql -U private_ai_stack private_ai_stack
