@echo off
setlocal

cd /d "%~dp0"

where docker >nul 2>nul
if errorlevel 1 (
    echo Docker was not found in PATH.
    echo Install Docker Desktop and enable GPU support before running this launcher.
    pause
    exit /b 1
)

if not exist docker.env (
    copy docker.env.example docker.env >nul
    echo Created docker.env from docker.env.example.
    echo Edit docker.env if you need another port, model, or Hugging Face token.
)

docker compose --env-file docker.env up --build
if errorlevel 1 (
    echo Docker Compose finished with an error.
    pause
    exit /b 1
)

endlocal
