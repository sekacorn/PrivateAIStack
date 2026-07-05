$ErrorActionPreference = "Stop"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ready" | ConvertTo-Json -Depth 8
