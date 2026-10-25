@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
 set "PY=.venv\Scripts\python.exe"
) else (
 set "PY=python"
)
%PY% ".\phase5_final_audit.py" --contract ".\phase5_audit_contract.json"
echo.
echo Read phase5_results\phase5_scientific_report_EN.md and phase5_results\phase5_audit_status.json
pause
