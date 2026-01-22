@echo off
REM Open frontend in default browser
echo ============================================
echo Opening Frontend in Browser
echo ============================================
echo.

REM Check if frontend exists
if not exist "frontend\index.html" (
    echo [ERROR] Frontend not found!
    pause
    exit /b 1
)

REM Open in default browser
start "" "frontend\index.html"

echo Frontend opened in browser!
echo Make sure backend server is running (start_backend.bat)
pause
