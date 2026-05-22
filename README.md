# Diarization System

Local Gradio service for Russian audio/video transcription, browser audio capture,
speaker diarization, summarization, and meeting protocol generation.

## Features

- Audio and video transcription with `faster-whisper`
- Speaker diarization with `pyannote.audio`
- Browser tab/window audio capture for videos that cannot be downloaded
- Summaries and protocols with Gemma
- Editable prompts for summaries and protocols in the UI
- Model settings page for Whisper/Gemma, including CPU/GPU selection for Whisper
- Local Hugging Face token storage in `model_cache/hf_token.txt`

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Open:

```text
http://127.0.0.1:3002
```

## Docker

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:3001
```

## Hugging Face Access

Diarization and Gemma can require a Hugging Face token. Create a read token,
paste it into the UI, and click `Сохранить токен`.

For diarization, the default model is:

```text
pyannote-community/speaker-diarization-community-1
```

Model cache, uploaded files, browser capture chunks, output files, and tokens are
ignored by git.

## Model Settings

Open `Настройки моделей` in the UI to change:

- Whisper model name
- Whisper device: `auto`, `cuda`, or `cpu`
- Whisper compute type and batch size
- Gemma/Hugging Face model id
- summary chunk size

Settings are stored locally in:

```text
model_cache/model_settings.json
```

