#!/bin/bash
# Smart Attendance System - Quick Start

echo "========================================"
echo " Smart Attendance System"
echo "========================================"

echo "[1/3] Installing dependencies..."
pip3 install -r requirements.txt || { echo "Install failed!"; exit 1; }

echo "[2/3] Creating database..."
mkdir -p database
python3 init_db.py

echo "[3/3] Starting server..."
echo " Open http://localhost:5000 in your browser"
echo " Press Ctrl+C to stop"
python3 backend/app.py
