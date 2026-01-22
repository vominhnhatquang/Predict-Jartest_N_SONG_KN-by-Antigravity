@echo off
REM Training script for Windows
echo ============================================
echo AI Model Training - DATA_FPT.csv
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then activate: venv\Scripts\activate
    echo Then install: pip install -r backend\requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment and run training
call venv\Scripts\activate
python train_model.py

echo.
echo ============================================
echo Training complete! Check the output above.
echo ============================================
pause
