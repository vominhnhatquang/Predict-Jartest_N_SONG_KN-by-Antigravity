@echo off
echo ============================================
echo SETUP - AI Model Web Integration
echo ============================================
echo.

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.8+ from:
    echo - Microsoft Store, or
    echo - https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo [OK] Python found!
python --version

echo.
echo [2/3] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    echo [OK] Virtual environment created!
)

echo.
echo [3/3] Installing dependencies...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

echo.
echo ============================================
echo SETUP COMPLETE!
echo ============================================
echo.
echo Next steps:
echo 1. Train model: train.bat
echo 2. Start backend: start_backend.bat
echo 3. Open frontend: open_frontend.bat
echo.
echo For detailed instructions, see: HUONG_DAN.md
echo ============================================
pause
