@echo off
title RL Agent Retraining GUI Launcher
echo ======================================================
echo   STARTING THE RL TRAINING & OPTIMIZATION GUI...
echo ======================================================
cd /d "%~dp0"

set PYTHON_EXE=.\env\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found at .\env\Scripts\python.exe
    echo Please make sure the 'env' folder exists, or create one:
    echo   1. python -m venv env
    echo   2. .\env\Scripts\activate
    echo   3. pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON_EXE%" -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] The Python virtual environment at .\env failed to execute.
    echo Please recreate the virtual environment:
    echo   1. rmdir /s /q env
    echo   2. python -m venv env
    echo   3. .\env\Scripts\activate
    echo   4. pip install -r requirements.txt
    pause
    exit /b 1
)

"%PYTHON_EXE%" train_gui.py
exit

