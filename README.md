# Diarization System

Local Gradio service for Russian audio/video transcription, browser audio capture,
speaker diarization, summarization, and meeting protocol generation.

## Features

- Audio and video transcription with `faster-whisper`
- Speaker diarization with `pyannote.audio`
- Browser tab/window audio capture for videos that cannot be downloaded
- Summaries and protocols with a local Hugging Face LLM
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

On the first run it will create `.venv`, install dependencies, create a desktop
shortcut, start the local service, and open the browser. The first dependency
installation can take a long time because PyTorch and ML packages are large.

After that, use the desktop shortcut `Diarization System` or double-click
`start_windows.bat` again.

Open manually if needed:

```text
http://127.0.0.1:3002
```

## Manual Start

```bash
pip install -r requirements.txt
python main.py
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
- summary chunk size
- maximum new tokens for each LLM response

Settings are stored locally in:

```text
model_cache/model_settings.json
```
