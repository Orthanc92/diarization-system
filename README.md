# Diarization System

Local Gradio service for Russian audio/video transcription, browser audio capture,
speaker diarization, summarization, and meeting protocol generation.

## Features

- Audio and video transcription with `faster-whisper`
- Speaker diarization with `pyannote.audio`
- Browser tab/window audio capture for videos that cannot be downloaded
- Configurable browser audio segment duration for transcription/diarization
- Summaries and protocols with a local Hugging Face LLM
- Optional vLLM OpenAI-compatible backend for faster LLM generation
- Free-form text/transcript analysis with custom user instructions
- Editable prompts for summaries and protocols in the UI
- Model settings page for Whisper/LLM, including CPU/GPU selection for Whisper
- Recommended summary/protocol model presets by available VRAM
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

## Docker

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:3001
```

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
- summary/protocol Hugging Face model id or one of the recommended presets
- LLM backend: local `transformers` or external `vLLM` OpenAI API
- system prompt shared by summary, protocol, and free-form text tasks
- summary chunk size
- maximum new tokens for each LLM response

Settings are stored locally in:

```text
model_cache/model_settings.json
```

## Optional vLLM Backend

The default LLM backend is local `transformers`, which is the simplest option
for Windows. For faster summary/protocol generation on Linux, WSL2, Docker, or a
separate GPU server, you can run vLLM separately and switch the UI setting
`Backend LLM` to `vLLM OpenAI API`.

Example vLLM server:

```bash
vllm serve google/gemma-4-E4B-it --host 127.0.0.1 --port 8000 --dtype bfloat16 --trust-remote-code
```

Then set in `Настройки моделей`:

```text
Backend LLM: vLLM OpenAI API
vLLM base URL: http://127.0.0.1:8000/v1
vLLM model: google/gemma-4-E4B-it
```

If your vLLM server requires an API key, set it before launch:

```powershell
$env:VLLM_API_KEY="your-key"
.\start_windows.bat
```
