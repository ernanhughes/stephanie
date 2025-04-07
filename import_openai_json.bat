@echo off
REM Stephanie: Import OpenAI JSON Export
REM Usage: import_openai_json.bat "C:\path\to\conversations.json"

SET SCRIPT_DIR=%~dp0
SET JSON_PATH=%1

IF "%JSON_PATH%"=="" (
    echo Please provide the path to conversations.json as the first argument.
    exit /b 1
)

call %SCRIPT_DIR%venv\Scripts\activate.bat

echo Importing JSON from: %JSON_PATH%
python %SCRIPT_DIR%\app\import_openai_json.py "%JSON_PATH%"

echo JSON import complete.
Dollar