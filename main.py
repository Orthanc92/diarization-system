import os
import uuid
import hashlib
import threading
import time
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app_logging import LOG_FILE, get_logger, setup_logging
from model_functions import (
    DEFAULT_LLM_SYSTEM_PROMPT,
    DEFAULT_PROTOCOL_PROMPT,
    DEFAULT_SUMMARY_PROMPT,
    DEFAULT_TEXT_TASK_INSTRUCTION,
    LLM_BACKEND_TRANSFORMERS,
    LLM_BACKEND_VLLM,
    RECOMMENDED_SUMMARY_MODELS,
    get_model_settings,
    HF_TOKEN_FILE,
    process_text_with_instruction,
    summarize_text_with_prompt_optimized,
    transcribe_audio,
    transcribe_audio_live,
    update_model_settings,
)

setup_logging()
logger = get_logger(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output_files")
BROWSER_CAPTURE_DIR = Path("browser_capture_chunks")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
BROWSER_CAPTURE_DIR.mkdir(exist_ok=True)
BROWSER_CAPTURE_SEGMENT_MS = int(os.getenv("BROWSER_CAPTURE_SEGMENT_MS", "60000"))
BROWSER_CAPTURE_MIN_SEGMENT_MS = 5000
BROWSER_CAPTURE_MAX_SEGMENT_MS = 10 * 60 * 1000

BROWSER_CAPTURE_SESSIONS = {}
BROWSER_CAPTURE_LOCK = threading.Lock()
BROWSER_CAPTURE_TRANSCRIBE_LOCK = threading.Lock()
BROWSER_CAPTURE_MODE_TRANSCRIPTION = "transcription"
BROWSER_CAPTURE_MODE_DIARIZATION = "diarization"
SUMMARY_MODEL_CUSTOM_CHOICE = "Своя модель / ручной HF id"


def _summary_model_choice_label(model):
    return f"{model['vram']} - {model['name']} ({model['model_id']})"


SUMMARY_MODEL_CHOICE_TO_ID = {
    _summary_model_choice_label(model): model["model_id"]
    for model in RECOMMENDED_SUMMARY_MODELS
}


def _initial_summary_model_choice(model_id):
    model_id = (model_id or "").strip()
    for choice, recommended_model_id in SUMMARY_MODEL_CHOICE_TO_ID.items():
        if recommended_model_id == model_id:
            return choice
    return SUMMARY_MODEL_CUSTOM_CHOICE


def select_recommended_summary_model(choice, current_model_id, current_vllm_model_id):
    model_id = SUMMARY_MODEL_CHOICE_TO_ID.get(choice)
    if not model_id:
        return [current_model_id, current_vllm_model_id]
    return [model_id, model_id]


def _recommended_summary_models_markdown():
    rows = [
        "| Память | Модель | HF id | Когда выбирать |",
        "| --- | --- | --- | --- |",
    ]
    for model in RECOMMENDED_SUMMARY_MODELS:
        rows.append(
            f"| {model['vram']} | {model['name']} | `{model['model_id']}` | {model['note']} |"
        )
    return (
        "#### Рекомендации по LLM для саммари и протокола\n\n"
        "AWQ-пресеты рассчитаны на 4-bit загрузку через `transformers` или `vLLM`; "
        "остальные оценки даны с запасом для обычной загрузки через `transformers`. "
        "Фактическое потребление зависит от длины контекста, драйверов и того, загружен ли параллельно Whisper.\n\n"
        + "\n".join(rows)
    )


def _generation_token_recommendations_markdown():
    return (
        "#### Справка по длине ответа модели\n\n"
        "`Макс. токенов ответа` - это лимит генерации на один вызов LLM. "
        "Для длинных текстов он применяется к каждому фрагменту и к финальному объединению, "
        "поэтому большие значения заметно увеличивают время обработки и расход VRAM.\n\n"
        "| Значение | Для чего подходит |\n"
        "| --- | --- |\n"
        "| 512-800 | короткое саммари, список вопросов, быстрый анализ |\n"
        "| 1000-1500 | обычный протокол, структурный разбор, задачи и выводы |\n"
        "| 2000-3000 | подробный протокол, большой список вопросов, глубокий анализ |\n"
        "| 4000+ | только если модель и видеопамять выдерживают; возможны повторы и сильное замедление |"
    )


def _mask_token(token):
    token = (token or "").strip()
    if not token:
        return "Токен не задан."
    if len(token) <= 8:
        return "Токен сохранен."
    return f"Токен сохранен: ...{token[-4:]}"


def _load_saved_hf_token():
    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token:
        try:
            token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
    if token:
        os.environ["HF_TOKEN"] = token
        logger.info("Hugging Face token loaded from environment or local file")
    return token


def _store_hf_token(token):
    HF_TOKEN_FILE.parent.mkdir(exist_ok=True, parents=True)
    HF_TOKEN_FILE.write_text(token, encoding="utf-8")
    try:
        HF_TOKEN_FILE.chmod(0o600)
    except OSError:
        pass


def save_hf_token(token):
    token = (token or "").strip()
    if not token:
        return [
            "",
            gr.update(value=""),
            "Токен не сохранен: поле пустое.",
        ]

    os.environ["HF_TOKEN"] = token
    _store_hf_token(token)
    logger.info("Hugging Face token saved locally: %s", HF_TOKEN_FILE)
    return [
        token,
        gr.update(value=""),
        f"{_mask_token(token)} Он сохранен локально в {HF_TOKEN_FILE}.",
    ]


def clear_hf_token():
    os.environ.pop("HF_TOKEN", None)
    try:
        HF_TOKEN_FILE.unlink()
    except FileNotFoundError:
        pass
    logger.info("Hugging Face token cleared")
    return [
        "",
        gr.update(value=""),
        "Токен очищен из текущего процесса и локального файла.",
    ]


def _model_settings_markdown(settings=None):
    settings = settings or get_model_settings()
    cuda_status = "доступна" if settings["cuda_available"] else "недоступна"
    llm_model = (
        settings["vllm_model_name"]
        if settings["llm_backend"] == LLM_BACKEND_VLLM
        else settings["summary_model_name"]
    )
    vllm_line = (
        f"\n- vLLM endpoint: `{settings['vllm_base_url']}`."
        if settings["llm_backend"] == LLM_BACKEND_VLLM
        else ""
    )
    return (
        "**Текущие настройки моделей**\n\n"
        f"- Whisper: `{settings['whisper_model_name']}`; "
        f"режим `{settings['whisper_device']}` -> `{settings['whisper_resolved_device']}`; "
        f"compute `{settings['whisper_resolved_compute_type']}`; "
        f"batch `{settings['whisper_batch_size']}`.\n"
        f"- LLM backend: `{settings['llm_backend']}`; model `{llm_model}`; "
        f"размер чанка `{settings['summary_max_chunk_size']}` токенов; "
        f"ответ до `{settings['summary_max_new_tokens']}` новых токенов.\n"
        f"- Генерация: sampling `{settings['llm_do_sample']}`; "
        f"temperature `{settings['llm_temperature']}`; top_p `{settings['llm_top_p']}`; "
        f"top_k `{settings['llm_top_k']}`; repetition `{settings['llm_repetition_penalty']}`; "
        f"no_repeat_ngram `{settings['llm_no_repeat_ngram_size']}`.\n"
        f"{vllm_line}\n"
        f"- CUDA: {cuda_status}.\n"
        f"- Логи: `{LOG_FILE}`."
    )


def save_model_settings_ui(
    whisper_model_name,
    whisper_device,
    whisper_compute_type,
    whisper_batch_size,
    whisper_cpu_threads,
    whisper_num_workers,
    llm_backend,
    summary_model_name,
    vllm_base_url,
    vllm_model_name,
    llm_system_prompt,
    summary_max_chunk_size,
    summary_max_new_tokens,
    llm_do_sample,
    llm_temperature,
    llm_top_p,
    llm_top_k,
    llm_repetition_penalty,
    llm_no_repeat_ngram_size,
):
    try:
        settings = update_model_settings(
            whisper_model_name=whisper_model_name,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            whisper_batch_size=whisper_batch_size,
            whisper_cpu_threads=whisper_cpu_threads,
            whisper_num_workers=whisper_num_workers,
            llm_backend=llm_backend,
            summary_model_name=summary_model_name,
            vllm_base_url=vllm_base_url,
            vllm_model_name=vllm_model_name,
            llm_system_prompt=llm_system_prompt,
            summary_max_chunk_size=summary_max_chunk_size,
            summary_max_new_tokens=summary_max_new_tokens,
            llm_do_sample=llm_do_sample,
            llm_temperature=llm_temperature,
            llm_top_p=llm_top_p,
            llm_top_k=llm_top_k,
            llm_repetition_penalty=llm_repetition_penalty,
            llm_no_repeat_ngram_size=llm_no_repeat_ngram_size,
        )
        return [
            _model_settings_markdown(settings),
            "Настройки сохранены. Если модель уже была загружена, она будет перезагружена при следующем запуске задачи.",
        ]
    except Exception as exc:
        logger.exception("Model settings save failed")
        return [
            _model_settings_markdown(),
            f"Ошибка сохранения настроек: {exc}",
        ]


def create_file_with_uuid(text, directory=OUTPUT_DIR):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)

    file_path = directory / f"{uuid.uuid4()}.txt"
    file_path.write_text(text, encoding="utf-8")
    logger.info("Output file created: path=%s chars=%s", file_path.resolve(), len(text or ""))
    return str(file_path.resolve())


def _browser_session_snapshot(session_id):
    with BROWSER_CAPTURE_LOCK:
        session = BROWSER_CAPTURE_SESSIONS.get(session_id)
        if not session:
            return {
                "session_id": session_id,
                "status": "Сессия захвата не найдена.",
                "transcript": "",
                "chunks": 0,
                "running": False,
                "segment_ms": BROWSER_CAPTURE_SEGMENT_MS,
            }
        return {
            "session_id": session_id,
            "status": session["status"],
            "transcript": session["transcript"],
            "chunks": session["chunks"],
            "running": session["running"],
            "mode": session.get("mode", BROWSER_CAPTURE_MODE_TRANSCRIPTION),
            "segment_ms": session.get("segment_ms", BROWSER_CAPTURE_SEGMENT_MS),
        }


def _append_browser_transcript(session_id, text, status):
    with BROWSER_CAPTURE_LOCK:
        session = BROWSER_CAPTURE_SESSIONS.get(session_id)
        if not session:
            logger.warning("Browser capture session not found while appending transcript: %s", session_id)
            return
        text = (text or "").strip()
        chunk_number = session["chunks"] + 1
        mode = session.get("mode", BROWSER_CAPTURE_MODE_TRANSCRIPTION)
        if text and not text.startswith("Ошибка"):
            if session["transcript"]:
                session["transcript"] += "\n\n" if mode == BROWSER_CAPTURE_MODE_DIARIZATION else " "
            if mode == BROWSER_CAPTURE_MODE_DIARIZATION:
                session["transcript"] += f"[Сегмент {chunk_number}]\n{text}"
            else:
                session["transcript"] += text
            session["status"] = status
            logger.info(
                "Browser capture segment appended: session=%s chunk=%s mode=%s chars=%s",
                session_id,
                chunk_number,
                mode,
                len(text),
            )
        elif text.startswith("Ошибка"):
            session["errors"].append(text)
            session["status"] = text
            logger.warning("Browser capture segment returned error: session=%s error=%s", session_id, text)
        else:
            session["status"] = (
                "Сегмент обработан, речь не распознана. "
                "Проверьте, что выбрана вкладка с видео и включена передача аудио."
            )
        session["chunks"] += 1
        session["updated_at"] = time.time()


def _normalize_browser_capture_mode(mode):
    mode = (mode or BROWSER_CAPTURE_MODE_TRANSCRIPTION).strip()
    if mode == BROWSER_CAPTURE_MODE_DIARIZATION:
        return BROWSER_CAPTURE_MODE_DIARIZATION
    return BROWSER_CAPTURE_MODE_TRANSCRIPTION


def _normalize_browser_capture_segment_ms(segment_seconds):
    try:
        segment_seconds = float(segment_seconds)
    except (TypeError, ValueError):
        return BROWSER_CAPTURE_SEGMENT_MS

    segment_ms = int(segment_seconds * 1000)
    return max(
        BROWSER_CAPTURE_MIN_SEGMENT_MS,
        min(BROWSER_CAPTURE_MAX_SEGMENT_MS, segment_ms),
    )


def _transcribe_browser_capture_chunk(session_id, chunk_path):
    with BROWSER_CAPTURE_LOCK:
        session = BROWSER_CAPTURE_SESSIONS.get(session_id, {})
        mode = session.get("mode", BROWSER_CAPTURE_MODE_TRANSCRIPTION)
        min_speakers = session.get("min_speakers")
        max_speakers = session.get("max_speakers")
        diarization_model_path = session.get("diarization_model_path")

    with BROWSER_CAPTURE_TRANSCRIBE_LOCK:
        logger.info(
            "Processing browser capture chunk: session=%s path=%s mode=%s",
            session_id,
            chunk_path,
            mode,
        )
        text = transcribe_audio(
            audio_path=None,
            video_path=str(chunk_path),
            enable_diarization=mode == BROWSER_CAPTURE_MODE_DIARIZATION,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            diarization_model_path=diarization_model_path,
        )
    try:
        Path(chunk_path).unlink(missing_ok=True)
        logger.debug("Browser capture chunk removed: %s", chunk_path)
    except OSError:
        logger.exception("Could not remove browser capture chunk: %s", chunk_path)
        pass
    _append_browser_transcript(
        session_id,
        text,
        f"Обработано сегментов: {_browser_session_snapshot(session_id)['chunks'] + 1}",
    )
    return _browser_session_snapshot(session_id)


def load_browser_capture_text(session_id):
    session_id = (session_id or "").strip()
    if not session_id:
        return [
            gr.DownloadButton(visible=False),
            "Сначала запустите захват звука вкладки браузера.",
        ]
    snapshot = _browser_session_snapshot(session_id)
    text = snapshot["transcript"].strip()
    if not text:
        logger.info("Browser capture text requested but empty: session=%s", session_id)
        return [
            gr.DownloadButton(visible=False),
            snapshot["status"] or "Текст захвата пока пуст.",
        ]
    path = create_file_with_uuid(text)
    logger.info("Browser capture text exported: session=%s chars=%s", session_id, len(text))
    return [
        gr.DownloadButton(label="Скачать", value=path, visible=True),
        text,
    ]


def _browser_capture_widget_html():
    return """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <style>
    :root { color-scheme: light dark; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2937;
      background: transparent;
    }
    .wrap {
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 12px;
      background: #ffffff;
    }
    .title { margin: 0 0 8px; font-size: 16px; font-weight: 650; }
    .hint { margin: 0 0 12px; font-size: 13px; line-height: 1.45; color: #4b5563; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
    .settings {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) minmax(130px, 150px) 110px 110px minmax(220px, 2fr);
      gap: 8px;
      margin-bottom: 10px;
      align-items: end;
    }
    label { display: grid; gap: 4px; font-size: 12px; color: #4b5563; }
    select, input {
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 8px 10px;
      background: #ffffff;
      color: #111827;
      font: 14px/1.3 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button {
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 8px 12px;
      background: #f9fafb;
      color: #111827;
      cursor: pointer;
      font-weight: 600;
    }
    button.primary { background: #111827; color: #ffffff; border-color: #111827; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    #status { font-size: 13px; color: #374151; }
    textarea {
      width: 100%;
      min-height: 140px;
      box-sizing: border-box;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      padding: 10px;
      resize: vertical;
      font: 14px/1.45 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      background: #ffffff;
      color: #111827;
    }
    @media (prefers-color-scheme: dark) {
      body { color: #e5e7eb; }
      .wrap { background: #111827; border-color: #374151; }
      .hint, label, #status { color: #d1d5db; }
      button { background: #1f2937; color: #f9fafb; border-color: #4b5563; }
      button.primary { background: #2563eb; border-color: #2563eb; }
      select, input { background: #030712; color: #f9fafb; border-color: #374151; }
      textarea { background: #030712; color: #f9fafb; border-color: #374151; }
    }
    @media (max-width: 760px) {
      .settings { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <p class="title">Захват звука вкладки браузера</p>
    <p class="hint">
      Нажмите старт, выберите вкладку или окно с видео и включите передачу аудио.
      В Chrome/Edge для вкладки обычно есть галка "Поделиться аудио вкладки".
    </p>
    <div class="settings">
      <label>
        Режим обработки
        <select id="processingMode">
          <option value="transcription">Только транскрибация</option>
          <option value="diarization">Диаризация по спикерам</option>
        </select>
      </label>
      <label>
        Длина сегмента
        <select id="segmentSeconds">
          <option value="15">15 сек</option>
          <option value="30">30 сек</option>
          <option value="60">1 мин</option>
          <option value="120">2 мин</option>
          <option value="180">3 мин</option>
          <option value="300">5 мин</option>
        </select>
      </label>
      <label class="diarization-setting">
        Мин. спикеров
        <input id="minSpeakers" type="number" min="1" step="1" placeholder="auto" />
      </label>
      <label class="diarization-setting">
        Макс. спикеров
        <input id="maxSpeakers" type="number" min="1" step="1" placeholder="auto" />
      </label>
      <label class="diarization-setting">
        Модель диаризации
        <input id="diarizationModelPath" type="text" placeholder="пусто = pyannote-community/speaker-diarization-community-1" />
      </label>
    </div>
    <div class="row">
      <button id="start" class="primary" type="button">Начать захват</button>
      <button id="stop" type="button" disabled>Остановить</button>
      <button id="copy" type="button">Копировать текст</button>
    </div>
    <div id="status">Захват не запущен.</div>
    <textarea id="transcript" readonly placeholder="Здесь будет появляться транскрипция звука вкладки..."></textarea>
  </div>

  <script>
    const state = {
      sessionId: "",
      stream: null,
      audioStream: null,
      recorder: null,
      stopping: false,
      pollTimer: null,
      segmentMs: __BROWSER_CAPTURE_SEGMENT_MS__,
      transcript: "",
    };

    const startBtn = document.getElementById("start");
    const stopBtn = document.getElementById("stop");
    const copyBtn = document.getElementById("copy");
    const statusEl = document.getElementById("status");
    const transcriptEl = document.getElementById("transcript");
    const processingModeEl = document.getElementById("processingMode");
    const minSpeakersEl = document.getElementById("minSpeakers");
    const maxSpeakersEl = document.getElementById("maxSpeakers");
    const diarizationModelPathEl = document.getElementById("diarizationModelPath");
    const segmentSecondsEl = document.getElementById("segmentSeconds");
    const diarizationSettingEls = Array.from(document.querySelectorAll(".diarization-setting"));

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function setParentValue(elemId, value) {
      try {
        const parentDoc = window.parent.document;
        const root = parentDoc.getElementById(elemId);
        if (!root) return;
        const field = root.querySelector("textarea, input");
        if (!field) return;
        const win = window.parent;
        const proto = field.tagName === "TEXTAREA"
          ? win.HTMLTextAreaElement.prototype
          : win.HTMLInputElement.prototype;
        const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
        setter.call(field, value);
        field.dispatchEvent(new Event("input", { bubbles: true }));
        field.dispatchEvent(new Event("change", { bubbles: true }));
      } catch (_err) {
        // Parent sync is best-effort; the local preview still works.
      }
    }

    function syncTranscript(text) {
      state.transcript = text || "";
      transcriptEl.value = state.transcript;
      setParentValue("transcription_output", state.transcript);
    }

    function syncSessionId(sessionId) {
      state.sessionId = sessionId || "";
      setParentValue("browser_capture_session", state.sessionId);
    }

    function setSettingsDisabled(disabled) {
      [processingModeEl, segmentSecondsEl, minSpeakersEl, maxSpeakersEl, diarizationModelPathEl]
        .forEach((elem) => { elem.disabled = disabled; });
    }

    function applyDefaultSegmentDuration() {
      const defaultSeconds = Math.max(5, Math.round(state.segmentMs / 1000));
      const value = String(defaultSeconds);
      if (!Array.from(segmentSecondsEl.options).some((option) => option.value === value)) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = `${defaultSeconds} сек`;
        segmentSecondsEl.appendChild(option);
      }
      segmentSecondsEl.value = value;
    }

    function toggleDiarizationSettings() {
      const enabled = processingModeEl.value === "diarization";
      diarizationSettingEls.forEach((elem) => {
        elem.style.display = enabled ? "grid" : "none";
      });
      transcriptEl.placeholder = enabled
        ? "Здесь будет появляться диаризованная транскрипция звука вкладки..."
        : "Здесь будет появляться транскрипция звука вкладки...";
    }

    function preferredMimeType() {
      const types = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "video/webm;codecs=opus",
        "video/webm"
      ];
      return types.find((type) => MediaRecorder.isTypeSupported(type)) || "";
    }

    async function startSession() {
      const form = new FormData();
      form.append("mode", processingModeEl.value);
      form.append("min_speakers", minSpeakersEl.value || "");
      form.append("max_speakers", maxSpeakersEl.value || "");
      form.append("diarization_model_path", diarizationModelPathEl.value || "");
      form.append("segment_seconds", segmentSecondsEl.value || "");
      const response = await fetch("/api/browser-capture/start", {
        method: "POST",
        body: form,
      });
      if (!response.ok) throw new Error("Не удалось создать сессию захвата.");
      const data = await response.json();
      syncSessionId(data.session_id);
      state.segmentMs = data.segment_ms || state.segmentMs;
      applyDefaultSegmentDuration();
      syncTranscript("");
      setStatus(data.status || "Захват запущен.");
    }

    async function sendChunk(blob) {
      if (!state.sessionId || !blob || blob.size === 0) return;
      const form = new FormData();
      form.append("session_id", state.sessionId);
      form.append("file", blob, `browser-audio-${Date.now()}.webm`);
      const response = await fetch("/api/browser-capture/chunk", {
        method: "POST",
        body: form,
      });
      if (!response.ok) throw new Error("Сервер не принял аудиосегмент.");
      const data = await response.json();
      syncTranscript(data.transcript || "");
      setStatus(data.status || "Сегмент обработан.");
    }

    function recordNextSegment() {
      if (state.stopping || !state.audioStream) return;

      const chunks = [];
      const mimeType = preferredMimeType();
      state.recorder = new MediaRecorder(
        state.audioStream,
        mimeType ? { mimeType } : undefined
      );

      state.recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) chunks.push(event.data);
      };

      state.recorder.onerror = (event) => {
        setStatus(`Ошибка записи аудио: ${event.error?.message || event.type}`);
      };

      state.recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: state.recorder.mimeType || "audio/webm" });
        try {
          if (blob.size > 0) {
            setStatus("Отправляю аудиосегмент на распознавание...");
            await sendChunk(blob);
          }
        } catch (err) {
          setStatus(err.message || String(err));
        }
        if (!state.stopping) {
          window.setTimeout(recordNextSegment, 250);
        }
      };

      state.recorder.start();
      window.setTimeout(() => {
        if (state.recorder && state.recorder.state === "recording") {
          state.recorder.stop();
        }
      }, state.segmentMs);
    }

    async function pollStatus() {
      if (!state.sessionId) return;
      try {
        const response = await fetch(`/api/browser-capture/status/${state.sessionId}`);
        if (!response.ok) return;
        const data = await response.json();
        syncTranscript(data.transcript || "");
        setStatus(data.status || "Захват идет...");
      } catch (_err) {
        // Polling is auxiliary; chunk uploads also update the preview.
      }
    }

    async function startCapture() {
      if (!navigator.mediaDevices?.getDisplayMedia) {
        setStatus("Браузер не поддерживает захват вкладки/экрана.");
        return;
      }

      startBtn.disabled = true;
      setSettingsDisabled(true);
      try {
        await startSession();
        state.stopping = false;
        state.stream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true,
        });
        const audioTracks = state.stream.getAudioTracks();
        if (!audioTracks.length) {
          throw new Error("Аудиодорожка не выбрана. Повторите старт и включите передачу аудио вкладки.");
        }
        state.audioStream = new MediaStream(audioTracks);
        state.stream.getTracks().forEach((track) => {
          track.onended = stopCapture;
        });
        stopBtn.disabled = false;
        setStatus(
          processingModeEl.value === "diarization"
            ? `Захват идет. Диаризация обновляется сегментами по ${Math.round(state.segmentMs / 1000)} сек.`
            : `Захват идет. Транскрипция обновляется сегментами по ${Math.round(state.segmentMs / 1000)} сек.`
        );
        state.pollTimer = window.setInterval(pollStatus, 2000);
        recordNextSegment();
      } catch (err) {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        setSettingsDisabled(false);
        setStatus(err.message || String(err));
      }
    }

    async function stopCapture() {
      state.stopping = true;
      stopBtn.disabled = true;
      startBtn.disabled = false;
      setSettingsDisabled(false);

      if (state.recorder && state.recorder.state === "recording") {
        state.recorder.stop();
      }
      if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
      }
      if (state.audioStream) {
        state.audioStream.getTracks().forEach((track) => track.stop());
      }
      if (state.pollTimer) {
        window.clearInterval(state.pollTimer);
      }

      if (state.sessionId) {
        const form = new FormData();
        form.append("session_id", state.sessionId);
        try {
          const response = await fetch("/api/browser-capture/stop", {
            method: "POST",
            body: form,
          });
          if (response.ok) {
            const data = await response.json();
            syncTranscript(data.transcript || state.transcript);
            setStatus(data.status || "Захват остановлен.");
          }
        } catch (_err) {
          setStatus("Захват остановлен.");
        }
      } else {
        setStatus("Захват остановлен.");
      }
    }

    startBtn.addEventListener("click", startCapture);
    stopBtn.addEventListener("click", stopCapture);
    processingModeEl.addEventListener("change", toggleDiarizationSettings);
    applyDefaultSegmentDuration();
    toggleDiarizationSettings();
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(state.transcript || transcriptEl.value || "");
        setStatus("Текст скопирован.");
      } catch (_err) {
        transcriptEl.select();
        setStatus("Скопируйте выделенный текст вручную.");
      }
    });
  </script>
</body>
</html>
""".replace("__BROWSER_CAPTURE_SEGMENT_MS__", str(BROWSER_CAPTURE_SEGMENT_MS))


def transcribe_and_create_file(
    audio_path,
    video_path,
    enable_diarization=False,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    diarization_model_path=None,
):
    logger.info(
        "UI transcription started: audio=%s video=%s diarization=%s",
        bool(audio_path),
        bool(video_path),
        bool(enable_diarization),
    )
    result = transcribe_audio(
        audio_path=audio_path,
        video_path=video_path,
        enable_diarization=enable_diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        diarization_model_path=diarization_model_path,
    )
    path = create_file_with_uuid(result)
    logger.info("UI transcription finished: output_chars=%s", len(result or ""))
    return [gr.DownloadButton(label="Скачать", value=path, visible=True), result]


def transcribe_video_on_play(
    auto_transcribe,
    video_path,
    enable_diarization=False,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    diarization_model_path=None,
    last_live_key=None,
):
    if not auto_transcribe or not video_path:
        yield [gr.update(), gr.update(), last_live_key]
        return

    token_fingerprint = (
        hashlib.sha256(hf_token.encode("utf-8")).hexdigest()[:12]
        if hf_token
        else ""
    )
    live_key = repr(
        (
            str(video_path),
            bool(enable_diarization),
            min_speakers,
            max_speakers,
            token_fingerprint,
            diarization_model_path,
        )
    )
    if live_key == last_live_key:
        logger.info("Skipped duplicate live video transcription: video=%s", video_path)
        yield [gr.update(), gr.update(), last_live_key]
        return

    logger.info(
        "Live video transcription started: video=%s diarization=%s",
        video_path,
        bool(enable_diarization),
    )
    final_text = ""
    for partial_text in transcribe_audio_live(
        audio_path=None,
        video_path=video_path,
        enable_diarization=enable_diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        hf_token=hf_token,
        diarization_model_path=diarization_model_path,
    ):
        final_text = partial_text
        yield [
            gr.DownloadButton(visible=False),
            partial_text,
            live_key,
        ]

    path = create_file_with_uuid(final_text)
    logger.info("Live video transcription finished: output_chars=%s", len(final_text or ""))
    yield [
        gr.DownloadButton(label="Скачать", value=path, visible=True),
        final_text,
        live_key,
    ]


def process_and_create_file(
    text,
    make_protocol=True,
    summary_prompt=None,
    protocol_prompt=None,
):
    logger.info(
        "UI summary/protocol started: input_chars=%s make_protocol=%s",
        len(text or ""),
        make_protocol,
    )
    result = summarize_text_with_prompt_optimized(
        text,
        make_protocol=make_protocol,
        summary_prompt=summary_prompt,
        protocol_prompt=protocol_prompt,
    )
    path = create_file_with_uuid(result)
    logger.info("UI summary/protocol finished: output_chars=%s", len(result or ""))
    return [gr.DownloadButton(label="Скачать", value=path, visible=True), result]


def process_text_task_and_create_file(text, instruction):
    logger.info(
        "UI free-form text task started: input_chars=%s instruction_chars=%s",
        len(text or ""),
        len(instruction or ""),
    )
    result = process_text_with_instruction(text, instruction)
    path = create_file_with_uuid(result)
    logger.info("UI free-form text task finished: output_chars=%s", len(result or ""))
    return [gr.DownloadButton(label="Скачать", value=path, visible=True), result]


def test_file_create(text):
    path = create_file_with_uuid(text)
    return gr.DownloadButton(label="Скачать", value=path, visible=True)


INITIAL_HF_TOKEN = _load_saved_hf_token()
INITIAL_MODEL_SETTINGS = get_model_settings()


with gr.Blocks(title="Транскрибация, диаризация и суммаризация") as demo:
    gr.Markdown("# Транскрибация, диаризация и суммаризация")
    model_settings_status = gr.Markdown(_model_settings_markdown(INITIAL_MODEL_SETTINGS))
    live_transcription_key = gr.State(value=None)
    hf_token_state = gr.State(value=INITIAL_HF_TOKEN)

    with gr.Accordion("Доступ Hugging Face", open=not bool(INITIAL_HF_TOKEN)):
        hf_token_input = gr.Textbox(
            label="Hugging Face token",
            type="password",
            placeholder="Нужен для скачивания gated-моделей pyannote/Gemma из Hugging Face",
        )
        with gr.Row():
            save_hf_token_btn = gr.Button("Сохранить токен")
            clear_hf_token_btn = gr.Button("Очистить токен")
        hf_token_status = gr.Markdown(_mask_token(INITIAL_HF_TOKEN))

    with gr.Tab("Настройки моделей"):
        gr.Markdown(
            "Здесь можно выбрать модели и устройство. Настройки сохраняются локально в "
            "`model_cache/model_settings.json` и применяются к следующим задачам."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Whisper")
                whisper_model_input = gr.Textbox(
                    label="Модель Whisper",
                    value=INITIAL_MODEL_SETTINGS["whisper_model_name"],
                    placeholder="Например: large-v3, medium, small",
                )
                whisper_device_input = gr.Radio(
                    label="Устройство для Whisper",
                    choices=["auto", "cuda", "cpu"],
                    value=INITIAL_MODEL_SETTINGS["whisper_device"],
                    info="auto использует GPU при наличии CUDA, иначе CPU. CPU медленнее, но не требует видеопамяти.",
                )
                whisper_compute_input = gr.Dropdown(
                    label="Compute type",
                    choices=["auto", "float16", "int8_float16", "int8", "float32"],
                    value=INITIAL_MODEL_SETTINGS["whisper_compute_type"],
                    info="Для GPU обычно лучше auto/float16. Для CPU обычно лучше auto/int8.",
                )
                whisper_batch_input = gr.Number(
                    label="Batch size",
                    value=INITIAL_MODEL_SETTINGS["whisper_batch_size"],
                    precision=0,
                    info="Больше batch быстрее на GPU, но требует больше VRAM. Если ловите OOM, уменьшите.",
                )
                whisper_cpu_threads_input = gr.Number(
                    label="CPU threads",
                    value=INITIAL_MODEL_SETTINGS["whisper_cpu_threads"],
                    precision=0,
                    info="Используется faster-whisper при работе на CPU.",
                )
                whisper_num_workers_input = gr.Number(
                    label="Num workers",
                    value=INITIAL_MODEL_SETTINGS["whisper_num_workers"],
                    precision=0,
                    info="Количество worker-потоков faster-whisper.",
                )
                gr.Markdown("### LLM / суммаризация и протокол")
                llm_backend_input = gr.Radio(
                    label="Backend LLM",
                    choices=[
                        (LLM_BACKEND_TRANSFORMERS, LLM_BACKEND_TRANSFORMERS),
                        ("vLLM OpenAI API", LLM_BACKEND_VLLM),
                    ],
                    value=INITIAL_MODEL_SETTINGS["llm_backend"],
                    info="Transformers работает локально в этом приложении. vLLM использует отдельно запущенный OpenAI-compatible server.",
                )
                summary_model_recommendation = gr.Dropdown(
                    label="Рекомендованная модель по памяти",
                    choices=[SUMMARY_MODEL_CUSTOM_CHOICE]
                    + list(SUMMARY_MODEL_CHOICE_TO_ID.keys()),
                    value=_initial_summary_model_choice(
                        INITIAL_MODEL_SETTINGS["summary_model_name"]
                    ),
                    info="Выберите вариант, чтобы подставить HF id в поле ниже. Затем сохраните настройки.",
                )
                summary_model_input = gr.Textbox(
                    label="Transformers HF id модели",
                    value=INITIAL_MODEL_SETTINGS["summary_model_name"],
                    placeholder="Например: google/gemma-4-E4B-it",
                    info="Используется при backend `transformers`. Модель должна помещаться в видеопамять. Для gated-моделей нужен HF token.",
                )
                vllm_base_url_input = gr.Textbox(
                    label="vLLM base URL",
                    value=INITIAL_MODEL_SETTINGS["vllm_base_url"],
                    placeholder="http://127.0.0.1:8000/v1",
                    info="Используется при backend `vllm`. Запустите vLLM отдельно, например `vllm serve ... --port 8000`.",
                )
                vllm_model_input = gr.Textbox(
                    label="vLLM model",
                    value=INITIAL_MODEL_SETTINGS["vllm_model_name"],
                    placeholder="Например: google/gemma-4-E4B-it",
                    info="Имя модели, с которым поднят vLLM server.",
                )
                llm_system_prompt_input = gr.Textbox(
                    label="Системный промпт LLM",
                    value=INITIAL_MODEL_SETTINGS.get(
                        "llm_system_prompt",
                        DEFAULT_LLM_SYSTEM_PROMPT,
                    ),
                    lines=5,
                    info="Общие правила для всех LLM-задач: саммари, протокол и свободная работа с текстом.",
                )
                summary_chunk_input = gr.Number(
                    label="Размер чанка для саммаризации, токены",
                    value=INITIAL_MODEL_SETTINGS["summary_max_chunk_size"],
                    precision=0,
                    info="Меньше чанки стабильнее, но больше проходов модели. Обычно 3072-4096.",
                )
                summary_max_new_tokens_input = gr.Number(
                    label="Макс. токенов ответа модели",
                    value=INITIAL_MODEL_SETTINGS["summary_max_new_tokens"],
                    precision=0,
                    info="Лимит новых токенов на один ответ LLM. Увеличьте для подробных протоколов и анализа больших текстов.",
                )
                with gr.Accordion("Параметры генерации LLM", open=False):
                    llm_do_sample_input = gr.Checkbox(
                        label="Sampling / do_sample",
                        value=INITIAL_MODEL_SETTINGS["llm_do_sample"],
                        info="Если выключено, модель отвечает более детерминированно. Если включено, temperature/top_p/top_k начинают сильнее влиять на вариативность.",
                    )
                    with gr.Row():
                        llm_temperature_input = gr.Number(
                            label="Temperature",
                            value=INITIAL_MODEL_SETTINGS["llm_temperature"],
                            precision=2,
                            info="Случайность ответа: 0-0.2 строже, 0.3-0.7 живее, выше 1 может быть менее стабильно.",
                        )
                        llm_top_p_input = gr.Number(
                            label="Top-p",
                            value=INITIAL_MODEL_SETTINGS["llm_top_p"],
                            precision=2,
                            info="Nucleus sampling: модель выбирает из наиболее вероятных токенов с суммарной вероятностью до этого значения. Обычно 0.7-0.95.",
                        )
                    with gr.Row():
                        llm_top_k_input = gr.Number(
                            label="Top-k",
                            value=INITIAL_MODEL_SETTINGS["llm_top_k"],
                            precision=0,
                            info="Ограничивает выбор k самыми вероятными токенами. 0 отключает ограничение.",
                        )
                        llm_repetition_penalty_input = gr.Number(
                            label="Repetition penalty",
                            value=INITIAL_MODEL_SETTINGS["llm_repetition_penalty"],
                            precision=2,
                            info="Штраф за повторы. 1.0 отключает штраф, 1.1-1.3 обычно уменьшает зацикливание.",
                        )
                    llm_no_repeat_ngram_input = gr.Number(
                        label="No repeat n-gram size",
                        value=INITIAL_MODEL_SETTINGS["llm_no_repeat_ngram_size"],
                        precision=0,
                        info="Запрещает повтор n-грамм указанной длины. 0 отключает, 4-8 помогает против повторов в длинных ответах.",
                    )
                    gr.Markdown(
                        "Рекомендация для протоколов: `do_sample` выключен, `temperature` 0.2-0.4, `top_p` 0.7-0.9, "
                        "`repetition_penalty` 1.1-1.25. Для творческого анализа включите sampling и поднимите temperature."
                    )
            with gr.Column():
                gr.Markdown(_recommended_summary_models_markdown())
                gr.Markdown(_generation_token_recommendations_markdown())
                gr.Markdown(
                    "Подсказка: на RTX 4090 для Whisper обычно выбирайте `auto`/`float16`. "
                    "Если параллельно загружена LLM и не хватает VRAM, временно переключите Whisper на `cpu` "
                    "или выберите LLM поменьше."
                )
        save_model_settings_btn = gr.Button("Сохранить настройки моделей", variant="primary")
        model_settings_save_status = gr.Markdown("")

    with gr.Tab("Транскрибация"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Загрузить аудио", type="filepath")
                video_input = gr.Video(
                    label="Загрузить или записать видео",
                    sources=["upload", "webcam"],
                    include_audio=True,
                )

                with gr.Accordion("Настройки диаризации", open=True):
                    diarization_input = gr.Checkbox(
                        label="Разделять речь по спикерам",
                        value=False,
                    )
                    with gr.Row():
                        min_speakers_input = gr.Number(
                            label="Мин. спикеров",
                            precision=0,
                        )
                        max_speakers_input = gr.Number(
                            label="Макс. спикеров",
                            precision=0,
                        )
                    diarization_model_path_input = gr.Textbox(
                        label="Локальный путь или HF id модели диаризации",
                        placeholder="Например: model_cache/pyannote/speaker-diarization",
                    )

                auto_video_transcribe = gr.Checkbox(
                    label="Потоково транскрибировать видео при запуске воспроизведения",
                    value=False,
                )
                transcribe_btn = gr.Button("Транскрибировать", variant="primary")

            with gr.Column():
                text_output = gr.Textbox(
                    label="Транскрибация",
                    lines=16,
                    elem_id="transcription_output",
                )
                download_transcription = gr.DownloadButton(visible=False)
                with gr.Row():
                    summarize_btn = gr.Button("Суммаризировать транскрибацию")
                    protocol_btn = gr.Button("Сделать протокол")
                with gr.Accordion("Промпты суммаризации и протокола", open=False):
                    gr.Markdown(
                        "Можно использовать `{summary}` или `{text}`. "
                        "Если плейсхолдер не указан, материал будет добавлен в конец промпта."
                    )
                    summary_prompt_input = gr.Textbox(
                        label="Промпт для суммаризации",
                        value=DEFAULT_SUMMARY_PROMPT,
                        lines=9,
                    )
                    protocol_prompt_input = gr.Textbox(
                        label="Промпт для протокола",
                        value=DEFAULT_PROTOCOL_PROMPT,
                        lines=10,
                    )
                summary_output = gr.Textbox(label="Результат", lines=8)
                download_result = gr.DownloadButton(visible=False)

    with gr.Tab("Суммаризация"):
        gr.Markdown("# Суммаризация и протоколирование")
        with gr.Column():
            text_input_tab2 = gr.Textbox(label="Текст", lines=8)
            with gr.Row():
                summarize_btn_tab2 = gr.Button("Суммаризировать")
                protocol_btn_tab2 = gr.Button("Создать протокол")
            with gr.Accordion("Промпты суммаризации и протокола", open=False):
                gr.Markdown(
                    "Можно использовать `{summary}` или `{text}`. "
                    "Если плейсхолдер не указан, материал будет добавлен в конец промпта."
                )
                summary_prompt_tab2 = gr.Textbox(
                    label="Промпт для суммаризации",
                    value=DEFAULT_SUMMARY_PROMPT,
                    lines=9,
                )
                protocol_prompt_tab2 = gr.Textbox(
                    label="Промпт для протокола",
                    value=DEFAULT_PROTOCOL_PROMPT,
                    lines=10,
                )
            text_tab2_output = gr.Textbox(label="Результат", lines=8)
            download_result_tab2 = gr.DownloadButton(visible=False)

    with gr.Tab("Работа с текстом"):
        gr.Markdown("# Свободная работа с текстом")
        gr.Markdown(
            "Вставьте транскрипцию или другой текст и напишите задачу для выбранной LLM. "
            "Можно попросить анализ, список вопросов, план, проверку аргументов, выделение задач или любой другой формат ответа."
        )
        with gr.Row():
            with gr.Column():
                text_task_input = gr.Textbox(
                    label="Текст / транскрипция",
                    lines=16,
                    placeholder="Вставьте сюда транскрипцию или нажмите кнопку ниже, чтобы взять текст из вкладки транскрибации.",
                )
                load_text_task_from_transcription_btn = gr.Button(
                    "Взять текст из транскрибации"
                )
            with gr.Column():
                text_task_instruction = gr.Textbox(
                    label="Что сделать с текстом",
                    value=DEFAULT_TEXT_TASK_INSTRUCTION,
                    lines=8,
                    placeholder="Например: составь 10 вопросов по тексту; найди спорные утверждения; сделай план действий.",
                )
                run_text_task_btn = gr.Button("Выполнить задачу", variant="primary")
                text_task_output = gr.Textbox(label="Ответ модели", lines=16)
                download_text_task = gr.DownloadButton(visible=False)

    with gr.Tab("Захват из браузера"):
        gr.Markdown(
            "Используйте этот режим, когда видео нельзя скачать. "
            "Браузер будет передавать звук выбранной вкладки или окна локальному серверу."
        )
        gr.HTML(
            '<iframe src="/browser-capture-widget" '
            'allow="display-capture; microphone; camera" '
            'style="width:100%; min-height:460px; border:0; border-radius:8px;"></iframe>'
        )
        browser_capture_session = gr.Textbox(
            label="ID сессии захвата",
            visible="hidden",
            elem_id="browser_capture_session",
        )
        sync_browser_capture_btn = gr.Button(
            "Загрузить текст захвата в поле транскрибации",
            variant="primary",
        )

    transcribe_inputs = [
        audio_input,
        video_input,
        diarization_input,
        min_speakers_input,
        max_speakers_input,
        hf_token_state,
        diarization_model_path_input,
    ]

    transcribe_btn.click(
        fn=transcribe_and_create_file,
        inputs=transcribe_inputs,
        outputs=[download_transcription, text_output],
    )

    video_input.play(
        fn=transcribe_video_on_play,
        inputs=[
            auto_video_transcribe,
            video_input,
            diarization_input,
            min_speakers_input,
            max_speakers_input,
            hf_token_state,
            diarization_model_path_input,
            live_transcription_key,
        ],
        outputs=[download_transcription, text_output, live_transcription_key],
    )

    sync_browser_capture_btn.click(
        fn=load_browser_capture_text,
        inputs=[browser_capture_session],
        outputs=[download_transcription, text_output],
    )

    save_hf_token_btn.click(
        fn=save_hf_token,
        inputs=[hf_token_input],
        outputs=[hf_token_state, hf_token_input, hf_token_status],
    )

    clear_hf_token_btn.click(
        fn=clear_hf_token,
        inputs=[],
        outputs=[hf_token_state, hf_token_input, hf_token_status],
    )

    summary_model_recommendation.change(
        fn=select_recommended_summary_model,
        inputs=[summary_model_recommendation, summary_model_input, vllm_model_input],
        outputs=[summary_model_input, vllm_model_input],
    )

    save_model_settings_btn.click(
        fn=save_model_settings_ui,
        inputs=[
            whisper_model_input,
            whisper_device_input,
            whisper_compute_input,
            whisper_batch_input,
            whisper_cpu_threads_input,
            whisper_num_workers_input,
            llm_backend_input,
            summary_model_input,
            vllm_base_url_input,
            vllm_model_input,
            llm_system_prompt_input,
            summary_chunk_input,
            summary_max_new_tokens_input,
            llm_do_sample_input,
            llm_temperature_input,
            llm_top_p_input,
            llm_top_k_input,
            llm_repetition_penalty_input,
            llm_no_repeat_ngram_input,
        ],
        outputs=[model_settings_status, model_settings_save_status],
    )

    summarize_btn.click(
        fn=lambda text, summary_prompt, protocol_prompt: process_and_create_file(
            text,
            False,
            summary_prompt,
            protocol_prompt,
        ),
        inputs=[text_output, summary_prompt_input, protocol_prompt_input],
        outputs=[download_result, summary_output],
    )

    protocol_btn.click(
        fn=lambda text, summary_prompt, protocol_prompt: process_and_create_file(
            text,
            True,
            summary_prompt,
            protocol_prompt,
        ),
        inputs=[text_output, summary_prompt_input, protocol_prompt_input],
        outputs=[download_result, summary_output],
    )

    summarize_btn_tab2.click(
        fn=lambda text, summary_prompt, protocol_prompt: process_and_create_file(
            text,
            False,
            summary_prompt,
            protocol_prompt,
        ),
        inputs=[text_input_tab2, summary_prompt_tab2, protocol_prompt_tab2],
        outputs=[download_result_tab2, text_tab2_output],
    )

    protocol_btn_tab2.click(
        fn=lambda text, summary_prompt, protocol_prompt: process_and_create_file(
            text,
            True,
            summary_prompt,
            protocol_prompt,
        ),
        inputs=[text_input_tab2, summary_prompt_tab2, protocol_prompt_tab2],
        outputs=[download_result_tab2, text_tab2_output],
    )

    load_text_task_from_transcription_btn.click(
        fn=lambda text: text,
        inputs=[text_output],
        outputs=[text_task_input],
    )

    run_text_task_btn.click(
        fn=process_text_task_and_create_file,
        inputs=[text_task_input, text_task_instruction],
        outputs=[download_text_task, text_task_output],
    )


def create_app(auth=None, server_name="127.0.0.1", server_port=3002):
    app = FastAPI()

    @app.get("/browser-capture-widget", response_class=HTMLResponse)
    async def browser_capture_widget():
        return HTMLResponse(_browser_capture_widget_html())

    @app.post("/api/browser-capture/start")
    async def browser_capture_start(
        mode: str = Form(BROWSER_CAPTURE_MODE_TRANSCRIPTION),
        min_speakers: str = Form(""),
        max_speakers: str = Form(""),
        diarization_model_path: str = Form(""),
        segment_seconds: str = Form(""),
    ):
        session_id = str(uuid.uuid4())
        now = time.time()
        mode = _normalize_browser_capture_mode(mode)
        segment_ms = _normalize_browser_capture_segment_ms(segment_seconds)
        status = (
            f"Сессия захвата с диаризацией создана. Длина сегмента: {segment_ms // 1000} сек."
            if mode == BROWSER_CAPTURE_MODE_DIARIZATION
            else f"Сессия захвата создана. Длина сегмента: {segment_ms // 1000} сек."
        )
        with BROWSER_CAPTURE_LOCK:
            BROWSER_CAPTURE_SESSIONS[session_id] = {
                "transcript": "",
                "status": status,
                "chunks": 0,
                "running": True,
                "errors": [],
                "mode": mode,
                "min_speakers": (min_speakers or "").strip(),
                "max_speakers": (max_speakers or "").strip(),
                "diarization_model_path": (diarization_model_path or "").strip(),
                "segment_ms": segment_ms,
                "created_at": now,
                "updated_at": now,
            }
        logger.info(
            "Browser capture session started: session=%s mode=%s segment_ms=%s min_speakers=%s max_speakers=%s model=%s",
            session_id,
            mode,
            segment_ms,
            (min_speakers or "").strip(),
            (max_speakers or "").strip(),
            (diarization_model_path or "").strip(),
        )
        return JSONResponse(_browser_session_snapshot(session_id))

    @app.post("/api/browser-capture/chunk")
    async def browser_capture_chunk(
        session_id: str = Form(...),
        file: UploadFile = File(...),
    ):
        session_id = session_id.strip()
        if not session_id:
            logger.warning("Browser capture chunk rejected: missing session id")
            return JSONResponse(
                {"status": "Не передан ID сессии.", "transcript": ""},
                status_code=400,
            )

        with BROWSER_CAPTURE_LOCK:
            session = BROWSER_CAPTURE_SESSIONS.get(session_id)
            if not session:
                logger.warning("Browser capture chunk rejected: session not found: %s", session_id)
                return JSONResponse(
                    {"status": "Сессия захвата не найдена.", "transcript": ""},
                    status_code=404,
                )
            mode = session.get("mode", BROWSER_CAPTURE_MODE_TRANSCRIPTION)
            session["status"] = (
                "Получен аудиосегмент, распознаю и разделяю спикеров..."
                if mode == BROWSER_CAPTURE_MODE_DIARIZATION
                else "Получен аудиосегмент, распознаю..."
            )

        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        chunk_path = BROWSER_CAPTURE_DIR / f"{session_id}-{uuid.uuid4()}{suffix}"
        file_content = await file.read()
        chunk_path.write_bytes(file_content)
        logger.info(
            "Browser capture chunk received: session=%s path=%s bytes=%s",
            session_id,
            chunk_path,
            len(file_content),
        )
        snapshot = await run_in_threadpool(
            _transcribe_browser_capture_chunk,
            session_id,
            chunk_path,
        )
        return JSONResponse(snapshot)

    @app.get("/api/browser-capture/status/{session_id}")
    async def browser_capture_status(session_id: str):
        return JSONResponse(_browser_session_snapshot(session_id))

    @app.post("/api/browser-capture/stop")
    async def browser_capture_stop(session_id: str = Form(...)):
        session_id = session_id.strip()
        with BROWSER_CAPTURE_LOCK:
            session = BROWSER_CAPTURE_SESSIONS.get(session_id)
            if session:
                session["running"] = False
                session["status"] = "Захват остановлен."
                session["updated_at"] = time.time()
                logger.info("Browser capture session stopped: session=%s chunks=%s", session_id, session["chunks"])
            else:
                logger.warning("Browser capture stop requested for missing session: %s", session_id)
        return JSONResponse(_browser_session_snapshot(session_id))

    return gr.mount_gradio_app(
        app,
        demo.queue(default_concurrency_limit=1),
        path="/",
        server_name=server_name,
        server_port=server_port,
        auth=auth,
        allowed_paths=[str(OUTPUT_DIR.resolve()), str(UPLOAD_DIR.resolve())],
    )


if __name__ == "__main__":
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "3002"))
    username = os.getenv("GRADIO_AUTH_USER", "")
    password = os.getenv("GRADIO_AUTH_PASSWORD", "")
    auth = (username, password) if username and password else None

    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    app = create_app(auth=auth, server_name=server_name, server_port=server_port)
    logger.info("Running on local URL: http://%s:%s", server_name, server_port)
    uvicorn.run(
        app,
        host=server_name,
        port=server_port,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )
