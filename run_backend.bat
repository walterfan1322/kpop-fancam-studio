@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
uvicorn backend.main:app --host 127.0.0.1 --port 8770 --reload
