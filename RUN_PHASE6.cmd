@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" ".\phase6_release.py" --contract ".\phase6_release_contract.json"
) else (
    python ".\phase6_release.py" --contract ".\phase6_release_contract.json"
)
if errorlevel 1 (
    echo Phase 6 evidence release FAILED. Check phase6_results\phase6_release_status.json
    pause
    exit /b 1
)
echo Phase 6 evidence package successfully generated in phase6_results.
pause
