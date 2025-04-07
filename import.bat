@echo off
call .\venv\Scripts\activate.bat
call venv\Scripts\activate.bat

set INPUT_PATH=C:\OpenAI\conversations.json

python stephanie/embed_and_store.py "%INPUT_PATH%"

