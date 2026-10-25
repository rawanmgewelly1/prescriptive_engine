@echo off
cd /d "%~dp0"
python run_pipeline.py --scenario configs/scenarios/RESEARCH_DEMO_200KW.json
if errorlevel 1 pause
