$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

# This process is deliberately loopback-only and uses the isolated chat DB.
$env:SIS_CONFIG_NAME = "sis_assistant_01"
$env:SIS_HOST = "127.0.0.1"
$env:SIS_PORT = "8766"
$env:SIS_ALLOWED_HOSTS = "127.0.0.1,localhost"
$env:SIS_ALLOWED_ORIGINS = "http://127.0.0.1:8766,http://localhost:8766"
$env:SIS_AUTH_DISABLED = "true"
$env:STEPHANIE_DB_NAME = "stephanie_assistant_01"

& (Join-Path $repoRoot "venv\Scripts\python.exe") -m uvicorn sis.main:app `
  --host 127.0.0.1 --port 8766
