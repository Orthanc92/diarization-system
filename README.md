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

Diarization and gated Hugging Face models can require a token. Create a read token,
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
- summary/protocol Hugging Face model id or one of the recommended presets
- summary chunk size
- maximum new tokens for each LLM response

Settings are stored locally in:

```text
model_cache/model_settings.json
```
