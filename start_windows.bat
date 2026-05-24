@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_windows.ps1" %*
if errorlevel 1 (
    echo.
    echo Startup failed. Check logs\app.log and logs\launcher-service.err.log.
    pause
)

endlocal
