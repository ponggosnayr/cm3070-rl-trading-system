@echo off
setlocal
cd /d "%~dp0"
title AI Financial Advisor
set "ADVISOR_PYTHON=%~dp0env\Scripts\python.exe"
"%ADVISOR_PYTHON%" -c "import sys" >nul 2>&1
if not errorlevel 1 goto launch
rem Local development fallback; examiners should install the project environment.
set "ADVISOR_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
"%ADVISOR_PYTHON%" -c "import sys" >nul 2>&1
if not errorlevel 1 goto launch
echo Project Python environment is missing or cannot start.
echo From this folder, run:
echo   python -m venv env
echo   env\Scripts\python -m pip install -r requirements.txt
echo   cd frontend ^&^& npm install
pause
exit /b 1
:launch
"%ADVISOR_PYTHON%" -X utf8 "%~dp0launch_dashboard.py" %*
if errorlevel 1 pause
