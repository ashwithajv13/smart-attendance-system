@echo off
title Smart Attendance System
color 0A

echo.
echo  ==========================================
echo   Smart Attendance System
echo  ==========================================
echo.

REM Use python -m pip instead of pip directly (works even if pip not on PATH)
echo [1/3] Installing dependencies...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo  ERROR: pip install failed.
    echo  Make sure Python is installed: https://python.org
    pause
    exit /b 1
)

echo [2/3] Setting up database...
if not exist "database" mkdir database
python init_db.py
if errorlevel 1 (
    echo  ERROR: Database setup failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting server...
echo.
echo  ==========================================
echo   Open this in your browser:
echo   http://localhost:5000
echo  ==========================================
echo.
echo  Press Ctrl+C to stop the server.
echo.

python backend/app.py

pause
