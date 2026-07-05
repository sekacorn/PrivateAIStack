$ErrorActionPreference = "Stop"
Invoke-RestMethod -Uri "http://127.0.0.1:8000/version" | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/models" | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/policies" | ConvertTo-Json -Depth 8
