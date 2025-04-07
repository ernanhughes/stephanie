@echo off
cd /d %~dp0
call venv\Scripts\activate.bat
start uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
