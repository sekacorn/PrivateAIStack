param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("memory", "audit")]
    [string]$Kind
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path exports | Out-Null

if ($Kind -eq "memory") {
    docker compose exec -T postgres psql -U private_ai_stack -d private_ai_stack -c "COPY (SELECT row_to_json(document_chunks) FROM document_chunks) TO STDOUT" |
        Set-Content -Encoding UTF8 "exports/memory.jsonl"
} else {
    docker compose cp api:/app/audit/audit.jsonl exports/audit.jsonl
}
