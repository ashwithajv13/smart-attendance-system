#!/usr/bin/env python3
"""Initialize the database — run once before starting the server."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Change to project root for relative paths to work
os.chdir(os.path.dirname(__file__))

from app import app, db

with app.app_context():
    db.create_all()
    print("✓ Database ready at database/attendance.db")
    print("  Run: python backend/app.py")
