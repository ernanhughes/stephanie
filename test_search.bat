@echo off
REM Batch file to test Stephanie's /api/search endpoint

set QUERY=freestyle
set MODE=hybrid

echo Sending search request...
curl -X POST http://127.0.0.1:8000/api/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"%QUERY%\", \"mode\": \"%MODE%\"}"
