@echo off
REM Stephanie: Import OpenAI HTML Chat Export
REM Usage: import_openai_html.bat "C:\path\to\messages.html"

SET SCRIPT_DIR=%~dp0
SET HTML_PATH=%1
echo Importing from: %HTML_PATH%



IF "%HTML_PATH%"=="" (
    echo ❌ Please provide the path to messages.html as the first argument.
    echo Example: import_openai_html.bat "C:\Users\You\Downloads\messages.html"
    exit /b 1
)

echo 🔄 Activating virtual environment...
call %SCRIPT_DIR%venv\Scripts\activate.bat

python %SCRIPT_DIR%\app\import_openai_html.py "%HTML_PATH%"

echo ✅ Import complete.
