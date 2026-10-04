@echo off
REM Double-click or run from PowerShell/CMD:  run.bat
REM Installs uv if needed, installs dependencies, starts the app and opens your browser.
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv ^(one-time^)...
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)
uv sync
if errorlevel 1 goto :fail
uv run python run.py
goto :eof
:fail
echo Setup failed. See SETUP.md ^(Troubleshooting^).
pause
