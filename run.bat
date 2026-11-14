@echo off
setlocal
cd /d "%~dp0"
if defined VIRTUAL_ENV (
    "%VIRTUAL_ENV%\Scripts\python.exe" -m manga_translator %*
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m manga_translator %*
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m manga_translator %*
) else (
    python -m manga_translator %*
)
exit /b %errorlevel%
