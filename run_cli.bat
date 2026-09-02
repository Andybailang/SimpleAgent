@echo off
setlocal enabledelayedexpansion

:: ========================================================================
:: Simple AI Agent CLI Launcher
:: ========================================================================

echo ==========================================
echo   Simple AI Agent CLI
echo ==========================================
echo.

:: Get script directory
cd /d "%~dp0"

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

:: Check/create virtual environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install dependencies
echo [INFO] Installing dependencies...
pip install -r requirements.txt --quiet

:: Run CLI
echo.
echo Starting Agent CLI...
echo Type /exit or /quit to quit
echo Type /clear to clear history
echo.
echo ==========================================
echo.

python cli.py

:: Exit with pause
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Program exited with error
)
pause
