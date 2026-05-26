FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128 && \
    grep -Ev '^(torch|torchaudio|gptqmodel|triton-windows)([<>=;[:space:]]|$)' requirements.txt > /tmp/requirements-docker.txt && \
    python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt

COPY . .

RUN mkdir -p uploads output_files browser_capture_chunks logs model_cache

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=3002
ENV LOG_DIR=/app/logs
ENV LOG_LEVEL=INFO
ENV HF_HOME=/app/model_cache/huggingface
ENV HF_HUB_CACHE=/app/model_cache/huggingface/hub
ENV MODEL_SETTINGS_FILE=/app/model_cache/docker_model_settings.json
ENV SUMMARY_MODEL_NAME=Qwen/Qwen3-14B-AWQ
ENV SUMMARY_MAX_CHUNK_SIZE=12000
ENV SUMMARY_MAX_NEW_TOKENS=2500
ENV LOCK_SERVICE_LLM_SETTINGS=1
ENV ALLOW_DOCKER_SHUTDOWN=1
ENV DOCKER_COMPOSE_PROJECT_NAME=diarization-system
ENV DOCKER_SOCKET_PATH=/var/run/docker.sock
ENV LLM_BACKEND=vllm
ENV VLLM_BASE_URL=http://vllm:8000/v1
ENV VLLM_MODEL_NAME=Qwen/Qwen3-14B-AWQ
ENV VLLM_CONTEXT_WINDOW=32768
ENV VLLM_CHUNK_CONTEXT_RESERVE_TOKENS=1024
ENV VLLM_CONTEXT_RETRY_RESERVE_TOKENS=16
ENV WHISPER_MODEL_NAME=large-v3
ENV WHISPER_DEVICE=cpu
ENV WHISPER_COMPUTE_TYPE=auto
ENV WHISPER_BATCH_SIZE=8
ENV WHISPER_CPU_THREADS=12
ENV WHISPER_NUM_WORKERS=6
ENV LIVE_TRANSCRIPTION_UPDATE_SECONDS=5
ENV BROWSER_CAPTURE_SEGMENT_MS=60000
ENV DIARIZATION_MODEL_NAME=pyannote-community/speaker-diarization-community-1
ENV PYANNOTE_METRICS_ENABLED=0
ENV HF_HUB_OFFLINE=0

EXPOSE 3002

CMD ["python", "main.py"]
