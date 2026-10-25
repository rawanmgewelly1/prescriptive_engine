@echo off
cd /d "%~dp0"
python run_dashboard.py
if errorlevel 1 pause
