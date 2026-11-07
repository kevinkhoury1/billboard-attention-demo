@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The project Python environment is missing. See the Phase 6 instructions in README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run "app\dashboard.py"
if errorlevel 1 pause
