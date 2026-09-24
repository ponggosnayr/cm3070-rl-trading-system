@echo off
cd /d "%~dp0"
echo ===================================================
echo   Running Unit Tests for RL Quant System
echo ===================================================
echo.

:: Use the Python inside the virtual environment specifically
set PYTHON_EXE=.\env\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Virtual environment not found at .\env
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
    echo This occurs when the project is moved to another computer where the
    echo base Python installation path differs from env\pyvenv.cfg.
    echo.
    echo Please recreate the virtual environment:
    echo   1. rmdir /s /q env
    echo   2. python -m venv env
    echo   3. .\env\Scripts\activate
    echo   4. pip install -r requirements.txt
    pause
    exit /b 1
)

:: Create logs folder if it doesn't exist
if not exist logs mkdir logs

:: Generate a portable timestamp for the log file using PowerShell
for /f "usebackq tokens=*" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMdd_HHmmss'"`) do set TIMESTAMP=%%i
set LOG_FILE=logs\test_run_%TIMESTAMP%.log

:: Run all tests in the tests/ directory and log output to the file while displaying it on the console
:: We add the 'src' folder to the PYTHONPATH so tests can find the modules
set PYTHONPATH=%PYTHONPATH%;%CD%\src

echo Logging results to: %LOG_FILE%
echo.

powershell -NoProfile -Command "$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'; & '%PYTHON_EXE%' -m pytest tests/ -v | Tee-Object -FilePath '%LOG_FILE%'"

echo.
echo ===================================================
echo   Tests Completed. Log saved to: %LOG_FILE%
echo ===================================================
pause
