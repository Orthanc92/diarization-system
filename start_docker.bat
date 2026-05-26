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

where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0launcher.py" --docker
    if errorlevel 1 (
        echo Docker launcher finished with an error.
        pause
        exit /b 1
    )
    exit /b 0
)

if not exist docker.env (
    copy docker.env.example docker.env >nul
    echo Created docker.env from docker.env.example.
    echo Edit docker.env if you need another port, model, or Hugging Face token.
)

docker compose --env-file docker.env up --build -d
if errorlevel 1 (
    echo Docker Compose finished with an error.
    pause
    exit /b 1
)

start "" "http://127.0.0.1:3002"
echo Docker service started in background.
echo Stop it from the UI or run: docker compose --env-file docker.env down

endlocal
