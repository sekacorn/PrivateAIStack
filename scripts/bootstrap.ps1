$ErrorActionPreference = "Stop"

foreach ($cmd in @("docker", "curl")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "Missing required command: $cmd"
    }
}

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env from .env.example"
} else {
    Write-Host ".env exists; not overwriting"
}

$model = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "qwen2.5:3b" }
Write-Host "Configured model: $model"
Write-Host "Model download is explicit. Run 'make model-pull' when you are ready."

docker compose up --build -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/health.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/smoke.ps1

Write-Host "API: http://127.0.0.1:8000"
Write-Host "Swagger: http://127.0.0.1:8000/docs"
Write-Host "Jaeger, when observability is enabled: http://127.0.0.1:16686"
