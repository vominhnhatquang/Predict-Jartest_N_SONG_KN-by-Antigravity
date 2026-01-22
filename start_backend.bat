@echo off
REM Start backend server
echo ============================================
echo Starting Flask Backend Server
echo ============================================
echo.

REM Check if model exists
if not exist "backend\model\trained_model.pkl" (
    echo [ERROR] Model not found!
    echo Please train the model first by running: train.bat
    pause
    exit /b 1
)

REM Activate virtual environment and start server
call venv\Scripts\activate
cd backend
python app.py

pause
