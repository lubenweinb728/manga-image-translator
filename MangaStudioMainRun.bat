@echo off
setlocal
cd /d "%~dp0"
if defined VIRTUAL_ENV (
    "%VIRTUAL_ENV%\Scripts\python.exe" MangaStudioMain.py %*
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" MangaStudioMain.py %*
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" MangaStudioMain.py %*
) else (
    python MangaStudioMain.py %*
)
exit /b %errorlevel%
