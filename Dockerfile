FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for uploads, downloads, browser chunks and model cache
RUN mkdir -p uploads output_files browser_capture_chunks model_cache

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=3001
ENV SUMMARY_MODEL_NAME=google/gemma-4-E4B-it
ENV SUMMARY_MAX_CHUNK_SIZE=4096
ENV LIVE_TRANSCRIPTION_UPDATE_SECONDS=5
ENV BROWSER_CAPTURE_SEGMENT_MS=60000
ENV DIARIZATION_MODEL_NAME=pyannote-community/speaker-diarization-community-1
ENV PYANNOTE_METRICS_ENABLED=0
ENV HF_HUB_OFFLINE=0
ENV HF_HOME=/app/model_cache
ENV HF_HUB_CACHE=/app/model_cache/hub

# Run the application
CMD ["python3", "main.py"] 
