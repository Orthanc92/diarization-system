# Diarization System

Local Gradio service for Russian audio/video transcription, browser audio capture,
speaker diarization, summarization, and meeting protocol generation.

## Features

- Audio and video transcription with `faster-whisper`
- Speaker diarization with `pyannote.audio`
- Browser tab/window audio capture for videos that cannot be downloaded
- Configurable browser audio segment duration for transcription/diarization
- Summaries and protocols with a local Hugging Face LLM or Docker vLLM backend
- vLLM OpenAI-compatible backend for faster LLM generation
- Free-form text/transcript analysis with custom user instructions
- Editable prompts for summaries and protocols in the UI
- Model settings page for Whisper, prompts, output length, and generation parameters
- Docker vLLM defaults fixed in `docker.env` to prevent accidental context/model changes
- Local Hugging Face token storage in `model_cache/hf_token.txt`

## Quick Start For Windows

1. Install Python 3.10+ from https://www.python.org/downloads/windows/.
   Enable `Add python.exe to PATH` during installation.
2. Download this repository as ZIP and unpack it.
3. Double-click:

```text
start_windows.bat
```

On the first run it will ask where to install the application. The recommended
folder is:

```text
%LOCALAPPDATA%\DiarizationSystem
```

After you choose the folder, the script copies a clean app copy there, creates
`.venv`, installs dependencies, creates a desktop shortcut, starts the local
service, and opens the browser. The first dependency installation can take a
long time because PyTorch and ML packages are large.
On Windows the startup script installs PyTorch `2.9.0` and torchaudio `2.9.0`
from the official CUDA 12.8 wheel index:

```text
https://download.pytorch.org/whl/cu128
```

To use another PyTorch wheel index, set `PYTORCH_INDEX_URL` before launch. For
CPU-only installation, use:

```powershell
$env:PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
.\start_windows.bat
```

After that, use the desktop shortcut `Diarization System` or double-click
`start_windows.bat` in the install folder again.

For portable mode, run from PowerShell:

```powershell
.\start_windows.bat -NoInstallPrompt
```

Open manually if needed:

```text
http://127.0.0.1:3002
```

## Manual Start

```bash
pip install -r requirements.txt
python main.py
```

For CUDA on manual Windows installs, install PyTorch from the official CUDA wheel
index first:

```powershell
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Windows App Launcher

The simplest launcher is `start_windows.bat`. If you also want a small exe
launcher, run once from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows_app.ps1
```

The script creates:

- `.venv` with project dependencies
- `dist\DiarizationSystem.exe`
- Desktop shortcut `Diarization System`

Double-click the shortcut to start the local service and open the browser.
Keep the launcher window open while using the service. Press `Ctrl+C` in that
window to stop it.

This exe is a launcher, not a monolithic bundle with PyTorch and all models
inside. It expects the project folder and `.venv` to stay next to it.

## Docker Desktop Launcher

For the Docker/vLLM path you can build a small launcher exe that does not bundle
models or Python dependencies. It starts Docker Compose in the background, waits
for the Gradio UI, and opens the browser:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_docker_app.ps1
```

The script creates:

- `.launcher-venv` with PyInstaller only
- `dist\DiarizationSystemDocker.exe`
- Desktop shortcut `Diarization System Docker`

Before launching it, start Docker Desktop. The first run can still take a long
time because Docker downloads images and model files. The UI includes
`Завершить Docker`, which stops the Docker Compose containers for this service.

## Docker + vLLM

Docker is the recommended path when you want faster LLM generation. The Compose
stack starts two containers:

- `vllm` - OpenAI-compatible vLLM server for summaries, protocols, and free-form text tasks
- `app` - Gradio application with Whisper, diarization, browser capture, and UI

Requirements:

- Docker Desktop with WSL2 backend
- NVIDIA driver and GPU support enabled in Docker Desktop
- Enough free VRAM for the selected vLLM model

Start on Windows:

```text
start_docker.bat
```

Or manually:

```powershell
copy docker.env.example docker.env
docker compose --env-file docker.env up --build -d
```

Open:

```text
http://127.0.0.1:3002
```

The vLLM API is also exposed on:

```text
http://127.0.0.1:8000/v1
```

The first launch downloads model files into:

```text
model_cache/huggingface
```

Docker defaults to `Qwen/Qwen3-14B-AWQ` through vLLM and keeps Whisper on CPU so
the LLM can use GPU memory. To change ports, model, Hugging Face token, or move
Whisper to GPU, edit `docker.env`. If port `3002` is already occupied, change
`APP_PORT`. For RTX 4090 the Docker default `VLLM_MAX_MODEL_LEN=32768` allows
larger text chunks while keeping a practical KV-cache reserve. The app container
also locks `SUMMARY_MAX_CHUNK_SIZE` from `docker.env`, so old UI-saved settings
cannot silently override the Docker vLLM context/chunk configuration.
`Qwen/Qwen3-32B-AWQ` is heavier and can fail on 24 GB GPUs because there is not
enough memory left for vLLM KV-cache.

Stop the stack:

```powershell
docker compose --env-file docker.env down
```

The Docker UI shutdown button uses the Docker socket mounted into the app
container. This is intended for a local desktop setup only; do not expose this
container to untrusted networks.

## Hugging Face Access

Diarization and gated Hugging Face models can require a token. Create a read token,
paste it into the UI, and click `Сохранить токен`.

For diarization, the default model is:

```text
pyannote-community/speaker-diarization-community-1
```

Model cache, uploaded files, browser capture chunks, output files, and tokens are
ignored by git.

## Logging

Application logs are written to:

```text
logs/app.log
```

The log file rotates automatically. You can change logging with environment
variables:

- `LOG_LEVEL`, default `INFO`
- `LOG_DIR`, default `logs`
- `LOG_FILE`, default `logs/app.log`
- `LOG_MAX_BYTES`, default `10485760`
- `LOG_BACKUP_COUNT`, default `5`

## Model Settings

Open `Настройки моделей` in the UI to change:

- Whisper model name
- Whisper device: `auto`, `cuda`, or `cpu`
- Whisper compute type and batch size
- system prompt shared by summary, protocol, and free-form text tasks
- maximum new tokens for each LLM response
- LLM generation parameters: sampling, temperature, top-p, top-k, repetition penalty, and no-repeat n-gram size

The UI intentionally does not expose vLLM backend/model/context/chunk settings.
For Docker runs they are fixed by `docker.env` and enforced in the app container:

- `VLLM_MODEL_NAME`
- `VLLM_MAX_MODEL_LEN`
- `VLLM_CHUNK_CONTEXT_RESERVE_TOKENS`
- `VLLM_CONTEXT_RETRY_RESERVE_TOKENS`
- `SUMMARY_MAX_CHUNK_SIZE`

Settings are stored locally in:

```text
model_cache/model_settings.json
```

## Optional External vLLM Backend

The recommended faster path is `start_docker.bat`, which starts the internal
vLLM server and locks the app to the Docker LLM settings. If you already have
vLLM on another machine, set the backend through environment variables before
starting the app.

Example external vLLM server:

```bash
vllm serve --model Qwen/Qwen3-14B-AWQ --served-model-name Qwen/Qwen3-14B-AWQ --host 0.0.0.0 --port 8000 --trust-remote-code --quantization awq --max-model-len 32768 --gpu-memory-utilization 0.90
```

Then start the app with:

```powershell
$env:LOCK_SERVICE_LLM_SETTINGS="1"
$env:LLM_BACKEND="vllm"
$env:VLLM_BASE_URL="http://127.0.0.1:8000/v1"
$env:VLLM_MODEL_NAME="Qwen/Qwen3-14B-AWQ"
$env:SUMMARY_MODEL_NAME="Qwen/Qwen3-14B-AWQ"
$env:VLLM_CONTEXT_WINDOW="32768"
$env:SUMMARY_MAX_CHUNK_SIZE="12000"
.\start_windows.bat
```

If your external vLLM server requires an API key, set it before launch:

```powershell
$env:VLLM_API_KEY="your-key"
.\start_windows.bat
```
