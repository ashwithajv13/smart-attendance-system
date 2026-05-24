@echo off
REM Smart Attendance System - Quick Start

echo ========================================
echo  Smart Attendance System
echo ========================================
echo.

echo [1/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 ( echo Install failed! & exit /b 1 )

echo.
echo [2/3] Creating database...
if not exist "database" mkdir database
python init_db.py

echo.
echo [3/3] Starting server...
echo  Open http://localhost:5000 in your browser
echo  Press Ctrl+C to stop
echo.
python backend/app.py
