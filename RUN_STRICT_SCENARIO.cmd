@echo off
cd /d "%~dp0"
python run_pipeline.py --scenario configs/scenarios/STRICT_NO_FC_BRIDGE.json
if errorlevel 1 pause
