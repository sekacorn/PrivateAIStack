Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$venv = ".audit-dev-venv"
$python = Join-Path $venv "Scripts/python.exe"

python -m bandit -r private_ai_stack

if (-not (Test-Path $python)) {
    python -m venv $venv
}

& $python -m pip install --upgrade pip pip-audit | Out-Host
& $python -m pip install -e ".[dev]" | Out-Host
& $python -m pip_audit --local
