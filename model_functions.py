import os
import re
import json
import subprocess
import tempfile
import gc
import warnings
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import torch

from app_logging import get_logger

logger = get_logger(__name__)

# Configuration
TEMPERATURE = 0.3
TOP_P = 0.7
TOP_K = 0
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 6
DO_SAMPLE = os.getenv("LLM_DO_SAMPLE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
NEW_TOKENS = int(os.getenv("SUMMARY_MAX_NEW_TOKENS", "800"))
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME", "google/gemma-4-E4B-it")
SUMMARY_MAX_CHUNK_SIZE = int(os.getenv("SUMMARY_MAX_CHUNK_SIZE", "4096"))
LLM_BACKEND_TRANSFORMERS = "transformers"
LLM_BACKEND_VLLM = "vllm"
LLM_BACKEND = os.getenv("LLM_BACKEND", LLM_BACKEND_TRANSFORMERS)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", SUMMARY_MODEL_NAME)
VLLM_TIMEOUT_SECONDS = float(os.getenv("VLLM_TIMEOUT_SECONDS", "300"))
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "32"))
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "12"))
WHISPER_NUM_WORKERS = int(os.getenv("WHISPER_NUM_WORKERS", "6"))
LIVE_TRANSCRIPTION_UPDATE_SECONDS = float(
    os.getenv("LIVE_TRANSCRIPTION_UPDATE_SECONDS", "5")
)
DIARIZATION_MODEL_NAME = os.getenv(
    "DIARIZATION_MODEL_NAME",
    "pyannote-community/speaker-diarization-community-1",
)
RECOMMENDED_SUMMARY_MODELS = [
    {
        "name": "Qwen3 1.7B",
        "model_id": "Qwen/Qwen3-1.7B",
        "vram": "6-8 GB / CPU",
        "note": "самый легкий вариант, если видеопамяти мало или нужна работа на CPU",
    },
    {
        "name": "Qwen3 4B",
        "model_id": "Qwen/Qwen3-4B",
        "vram": "8-12 GB",
        "note": "быстрый компромисс для обычных саммари и протоколов",
    },
    {
        "name": "Qwen3 8B",
        "model_id": "Qwen/Qwen3-8B",
        "vram": "16 GB",
        "note": "лучше держит длинный контекст и русскоязычные инструкции",
    },
    {
        "name": "Gemma 4 E4B IT",
        "model_id": "google/gemma-4-E4B-it",
        "vram": "24 GB",
        "note": "текущий рекомендуемый вариант для RTX 4090",
    },
    {
        "name": "Qwen3 14B",
        "model_id": "Qwen/Qwen3-14B",
        "vram": "32+ GB",
        "note": "более тяжелый вариант для машин с большим запасом видеопамяти",
    },
]
DEFAULT_SUMMARY_PROMPT = (
    "Ниже приведены частичные выжимки длинной русскоязычной транскрипции.\n"
    "Собери из них единое итоговое резюме.\n"
    "Правила: только русский язык; без HTML/XML-тегов; без испанского и английского; "
    "без выдуманных фактов; не упоминай номера сегментов и технические детали транскрибации.\n"
    "Формат ответа:\n"
    "1. Главная тема\n"
    "2. Ключевые тезисы\n"
    "3. Практические рекомендации / выводы\n\n"
    "Частичные выжимки:\n{summary}"
)
DEFAULT_PROTOCOL_PROMPT = (
    "Ниже приведен общий текст суммаризации нескольких частей встречи:\n\n"
    "{summary}\n\n"
    "Теперь сформируй протокол. Правила: только русский язык; без HTML/XML-тегов; "
    "без испанского и английского; без выдуманных фактов; не упоминай технические номера сегментов.\n"
    "Укажи следующие разделы:\n"
    "1. Повестка (основные темы)\n"
    "2. Подробно изложи основные обсуждения и принятые решения в семи предложениях\n"
    "3. Если есть, представь список задач / поручений\n"
    "Ответ подай в деловом стиле, подробно и структурированно."
)
DEFAULT_TEXT_TASK_INSTRUCTION = (
    "Проанализируй транскрипцию. Выдели ключевые темы, важные тезисы, "
    "открытые вопросы и возможные следующие шаги. Ответ дай структурированно."
)
DEFAULT_LLM_SYSTEM_PROMPT = (
    "Ты локальный помощник для анализа русскоязычных транскрипций. "
    "Отвечай на русском языке, опирайся только на переданный материал, "
    "не выдумывай факты и сохраняй структуру, которую просит пользователь."
)

# Cache
CACHE_DIR = Path.cwd() / "model_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
HF_TOKEN_FILE = Path(os.getenv("HF_TOKEN_FILE", str(CACHE_DIR / "hf_token.txt")))
MODEL_SETTINGS_FILE = Path(
    os.getenv("MODEL_SETTINGS_FILE", str(CACHE_DIR / "model_settings.json"))
)

hf_tokenizer = None
hf_model = None
hf_model_name = None
whisper_pipeline = None
whisper_pipeline_key = None
diarization_pipeline = None
diarization_pipeline_name = None
MODEL_SETTINGS = None


def _to_path(value):
    if not value:
        return None
    if isinstance(value, (str, os.PathLike)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return _to_path(value[0]) if value else None
    if isinstance(value, dict):
        for key in ("path", "name"):
            if value.get(key):
                return str(value[key])
        if value.get("video"):
            return _to_path(value["video"])
    return str(value)


def _optional_int(value):
    if value in (None, "", 0, 0.0):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value, default, minimum=1):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _non_negative_int(value, default):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _bounded_float(value, default, minimum=None, maximum=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on", "y", "да"}:
        return True
    if value in {"0", "false", "no", "off", "n", "нет"}:
        return False
    return default


def _normalize_llm_backend(value):
    value = (value or LLM_BACKEND_TRANSFORMERS).strip().lower()
    return value if value in {LLM_BACKEND_TRANSFORMERS, LLM_BACKEND_VLLM} else LLM_BACKEND_TRANSFORMERS


def _normalize_vllm_base_url(value):
    value = (value or VLLM_BASE_URL).strip().rstrip("/")
    return value or VLLM_BASE_URL


def _normalize_generation_settings(settings):
    settings = settings or {}
    return {
        "llm_do_sample": _to_bool(settings.get("llm_do_sample"), DO_SAMPLE),
        "llm_temperature": _bounded_float(
            settings.get("llm_temperature"),
            TEMPERATURE,
            minimum=0.0,
            maximum=2.0,
        ),
        "llm_top_p": _bounded_float(
            settings.get("llm_top_p"),
            TOP_P,
            minimum=0.01,
            maximum=1.0,
        ),
        "llm_top_k": _non_negative_int(settings.get("llm_top_k"), TOP_K),
        "llm_repetition_penalty": _bounded_float(
            settings.get("llm_repetition_penalty"),
            REPETITION_PENALTY,
            minimum=0.1,
            maximum=3.0,
        ),
        "llm_no_repeat_ngram_size": _non_negative_int(
            settings.get("llm_no_repeat_ngram_size"),
            NO_REPEAT_NGRAM_SIZE,
        ),
    }


def _default_model_settings():
    return {
        "whisper_model_name": WHISPER_MODEL_NAME,
        "whisper_device": _normalize_whisper_device(WHISPER_DEVICE),
        "whisper_compute_type": _normalize_whisper_compute_type(WHISPER_COMPUTE_TYPE),
        "whisper_batch_size": WHISPER_BATCH_SIZE,
        "whisper_cpu_threads": WHISPER_CPU_THREADS,
        "whisper_num_workers": WHISPER_NUM_WORKERS,
        "summary_model_name": SUMMARY_MODEL_NAME,
        "summary_max_chunk_size": SUMMARY_MAX_CHUNK_SIZE,
        "summary_max_new_tokens": NEW_TOKENS,
        "llm_backend": _normalize_llm_backend(LLM_BACKEND),
        "vllm_base_url": _normalize_vllm_base_url(VLLM_BASE_URL),
        "vllm_model_name": VLLM_MODEL_NAME,
        "llm_system_prompt": DEFAULT_LLM_SYSTEM_PROMPT,
        "llm_do_sample": DO_SAMPLE,
        "llm_temperature": TEMPERATURE,
        "llm_top_p": TOP_P,
        "llm_top_k": TOP_K,
        "llm_repetition_penalty": REPETITION_PENALTY,
        "llm_no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
    }


def _normalize_whisper_device(value):
    value = (value or "auto").strip().lower()
    return value if value in {"auto", "cuda", "cpu"} else "auto"


def _normalize_whisper_compute_type(value):
    value = (value or "auto").strip().lower()
    allowed = {"auto", "float16", "int8_float16", "int8", "float32"}
    return value if value in allowed else "auto"


def _resolve_whisper_device(device_mode):
    device_mode = _normalize_whisper_device(device_mode)
    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна. Выберите CPU или auto.")
        return "cuda"
    if device_mode == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_whisper_compute_type(compute_type, resolved_device):
    compute_type = _normalize_whisper_compute_type(compute_type)
    if compute_type != "auto":
        return compute_type
    return "float16" if resolved_device == "cuda" else "int8"


def _load_model_settings_from_disk():
    settings = _default_model_settings()
    try:
        saved_settings = json.loads(MODEL_SETTINGS_FILE.read_text(encoding="utf-8"))
        if isinstance(saved_settings, dict):
            settings.update(saved_settings)
    except FileNotFoundError:
        logger.info("Model settings file not found, using defaults: %s", MODEL_SETTINGS_FILE)
    except OSError as exc:
        logger.warning("Could not read model settings file %s: %s", MODEL_SETTINGS_FILE, exc)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid model settings JSON in %s: %s", MODEL_SETTINGS_FILE, exc)
        pass

    settings["whisper_model_name"] = (
        settings.get("whisper_model_name") or WHISPER_MODEL_NAME
    ).strip()
    settings["whisper_device"] = _normalize_whisper_device(
        settings.get("whisper_device")
    )
    settings["whisper_compute_type"] = _normalize_whisper_compute_type(
        settings.get("whisper_compute_type")
    )
    settings["whisper_batch_size"] = _positive_int(
        settings.get("whisper_batch_size"), WHISPER_BATCH_SIZE
    )
    settings["whisper_cpu_threads"] = _positive_int(
        settings.get("whisper_cpu_threads"), WHISPER_CPU_THREADS
    )
    settings["whisper_num_workers"] = _positive_int(
        settings.get("whisper_num_workers"), WHISPER_NUM_WORKERS
    )
    settings["summary_model_name"] = (
        settings.get("summary_model_name") or SUMMARY_MODEL_NAME
    ).strip()
    settings["summary_max_chunk_size"] = _positive_int(
        settings.get("summary_max_chunk_size"), SUMMARY_MAX_CHUNK_SIZE, minimum=256
    )
    settings["summary_max_new_tokens"] = _positive_int(
        settings.get("summary_max_new_tokens"), NEW_TOKENS, minimum=64
    )
    settings["llm_backend"] = _normalize_llm_backend(settings.get("llm_backend"))
    settings["vllm_base_url"] = _normalize_vllm_base_url(settings.get("vllm_base_url"))
    settings["vllm_model_name"] = (
        settings.get("vllm_model_name") or settings["summary_model_name"] or VLLM_MODEL_NAME
    ).strip()
    settings["llm_system_prompt"] = (
        settings.get("llm_system_prompt") or DEFAULT_LLM_SYSTEM_PROMPT
    ).strip()
    settings.update(_normalize_generation_settings(settings))
    return settings


def _save_model_settings_to_disk(settings):
    MODEL_SETTINGS_FILE.parent.mkdir(exist_ok=True, parents=True)
    MODEL_SETTINGS_FILE.write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Model settings saved to %s", MODEL_SETTINGS_FILE)


def get_model_settings():
    settings = dict(MODEL_SETTINGS)
    try:
        resolved_device = _resolve_whisper_device(settings["whisper_device"])
    except RuntimeError:
        resolved_device = "cpu"
    settings["cuda_available"] = torch.cuda.is_available()
    settings["whisper_resolved_device"] = resolved_device
    settings["whisper_resolved_compute_type"] = _resolve_whisper_compute_type(
        settings["whisper_compute_type"],
        resolved_device,
    )
    return settings


def unload_whisper_model():
    global whisper_pipeline, whisper_pipeline_key
    if whisper_pipeline is not None:
        logger.info("Unloading Whisper model")
    whisper_pipeline = None
    whisper_pipeline_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_hf_model():
    global hf_tokenizer, hf_model, hf_model_name
    if hf_model is not None or hf_tokenizer is not None:
        logger.info("Unloading LLM model: %s", hf_model_name)
    hf_tokenizer = None
    hf_model = None
    hf_model_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def update_model_settings(
    whisper_model_name=None,
    whisper_device=None,
    whisper_compute_type=None,
    whisper_batch_size=None,
    whisper_cpu_threads=None,
    whisper_num_workers=None,
    summary_model_name=None,
    summary_max_chunk_size=None,
    summary_max_new_tokens=None,
    llm_backend=None,
    vllm_base_url=None,
    vllm_model_name=None,
    llm_system_prompt=None,
    llm_do_sample=None,
    llm_temperature=None,
    llm_top_p=None,
    llm_top_k=None,
    llm_repetition_penalty=None,
    llm_no_repeat_ngram_size=None,
):
    global MODEL_SETTINGS

    old_settings = dict(MODEL_SETTINGS)
    new_settings = dict(old_settings)
    new_settings["whisper_model_name"] = (
        whisper_model_name or WHISPER_MODEL_NAME
    ).strip()
    new_settings["whisper_device"] = _normalize_whisper_device(whisper_device)
    new_settings["whisper_compute_type"] = _normalize_whisper_compute_type(
        whisper_compute_type
    )
    new_settings["whisper_batch_size"] = _positive_int(
        whisper_batch_size, old_settings["whisper_batch_size"]
    )
    new_settings["whisper_cpu_threads"] = _positive_int(
        whisper_cpu_threads, old_settings["whisper_cpu_threads"]
    )
    new_settings["whisper_num_workers"] = _positive_int(
        whisper_num_workers, old_settings["whisper_num_workers"]
    )
    new_settings["summary_model_name"] = (
        summary_model_name or SUMMARY_MODEL_NAME
    ).strip()
    new_settings["summary_max_chunk_size"] = _positive_int(
        summary_max_chunk_size,
        old_settings["summary_max_chunk_size"],
        minimum=256,
    )
    new_settings["summary_max_new_tokens"] = _positive_int(
        summary_max_new_tokens,
        old_settings["summary_max_new_tokens"],
        minimum=64,
    )
    new_settings["llm_backend"] = _normalize_llm_backend(llm_backend)
    new_settings["vllm_base_url"] = _normalize_vllm_base_url(vllm_base_url)
    new_settings["vllm_model_name"] = (
        vllm_model_name or new_settings["summary_model_name"] or VLLM_MODEL_NAME
    ).strip()
    new_settings["llm_system_prompt"] = (
        llm_system_prompt or DEFAULT_LLM_SYSTEM_PROMPT
    ).strip()
    new_settings.update(
        _normalize_generation_settings(
            {
                **old_settings,
                "llm_do_sample": (
                    llm_do_sample
                    if llm_do_sample is not None
                    else old_settings.get("llm_do_sample")
                ),
                "llm_temperature": (
                    llm_temperature
                    if llm_temperature is not None
                    else old_settings.get("llm_temperature")
                ),
                "llm_top_p": llm_top_p if llm_top_p is not None else old_settings.get("llm_top_p"),
                "llm_top_k": llm_top_k if llm_top_k is not None else old_settings.get("llm_top_k"),
                "llm_repetition_penalty": (
                    llm_repetition_penalty
                    if llm_repetition_penalty is not None
                    else old_settings.get("llm_repetition_penalty")
                ),
                "llm_no_repeat_ngram_size": (
                    llm_no_repeat_ngram_size
                    if llm_no_repeat_ngram_size is not None
                    else old_settings.get("llm_no_repeat_ngram_size")
                ),
            }
        )
    )

    _resolve_whisper_device(new_settings["whisper_device"])

    whisper_keys = {
        "whisper_model_name",
        "whisper_device",
        "whisper_compute_type",
        "whisper_batch_size",
        "whisper_cpu_threads",
        "whisper_num_workers",
    }
    summary_keys = {
        "summary_model_name",
        "summary_max_chunk_size",
        "summary_max_new_tokens",
        "llm_backend",
        "vllm_base_url",
        "vllm_model_name",
        "llm_system_prompt",
        "llm_do_sample",
        "llm_temperature",
        "llm_top_p",
        "llm_top_k",
        "llm_repetition_penalty",
        "llm_no_repeat_ngram_size",
    }

    MODEL_SETTINGS = new_settings
    _save_model_settings_to_disk(new_settings)
    logger.info(
        "Model settings updated: whisper=%s device=%s compute=%s batch=%s backend=%s llm=%s vllm=%s chunk=%s max_new_tokens=%s do_sample=%s temperature=%s top_p=%s top_k=%s repetition_penalty=%s no_repeat_ngram_size=%s",
        new_settings["whisper_model_name"],
        new_settings["whisper_device"],
        new_settings["whisper_compute_type"],
        new_settings["whisper_batch_size"],
        new_settings["llm_backend"],
        new_settings["summary_model_name"],
        new_settings["vllm_model_name"],
        new_settings["summary_max_chunk_size"],
        new_settings["summary_max_new_tokens"],
        new_settings["llm_do_sample"],
        new_settings["llm_temperature"],
        new_settings["llm_top_p"],
        new_settings["llm_top_k"],
        new_settings["llm_repetition_penalty"],
        new_settings["llm_no_repeat_ngram_size"],
    )

    if any(old_settings[key] != new_settings[key] for key in whisper_keys):
        unload_whisper_model()
    if any(old_settings[key] != new_settings[key] for key in summary_keys):
        unload_hf_model()

    return get_model_settings()


MODEL_SETTINGS = _load_model_settings_from_disk()


def _hf_token():
    env_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if env_token:
        return env_token.strip()

    try:
        return HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None

def _normalize_model_origin(model_origin):
    origin = (model_origin or DIARIZATION_MODEL_NAME).strip()
    origin = os.path.expandvars(os.path.expanduser(origin))
    return origin


def _resolve_local_model_origin(model_origin):
    origin = _normalize_model_origin(model_origin)
    path = Path(origin)
    if not path.exists():
        return origin, False

    if path.is_dir() and not (path / "config.yaml").exists():
        snapshots_dir = path / "snapshots"
        if snapshots_dir.exists():
            candidates = [
                snapshot
                for snapshot in snapshots_dir.iterdir()
                if snapshot.is_dir() and (snapshot / "config.yaml").exists()
            ]
            if candidates:
                newest_snapshot = max(candidates, key=lambda item: item.stat().st_mtime)
                return str(newest_snapshot), True

    return str(path), True


def _looks_like_local_path(model_origin):
    if not model_origin:
        return False
    path = Path(model_origin)
    return (
        path.is_absolute()
        or model_origin.startswith((".", "~", "/", "\\"))
        or bool(path.drive)
    )


def _format_timestamp(seconds):
    seconds = max(0, float(seconds or 0))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def remove_exact_sentence_repeats(text):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    cleaned = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and (not cleaned or cleaned[-1] != sentence):
            cleaned.append(sentence)
    return " ".join(cleaned)


def extract_audio_from_video(video_path):
    video_path = _to_path(video_path)
    if not video_path:
        raise ValueError("Не передан видеофайл.")

    logger.info("Extracting audio with ffmpeg: source=%s", video_path)
    fd, temp_filename = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        temp_filename,
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        stderr = result.stderr.strip()[-2000:]
        logger.error("ffmpeg audio extraction failed: source=%s stderr=%s", video_path, stderr)
        raise RuntimeError(f"Не удалось извлечь аудиодорожку из видео через ffmpeg: {stderr}")

    logger.info("Audio extracted: source=%s output=%s", video_path, temp_filename)
    return temp_filename


def get_audio_source(audio_path, video_path, force_wav=False):
    audio_path = _to_path(audio_path)
    video_path = _to_path(video_path)
    if audio_path:
        if force_wav:
            extracted_audio = extract_audio_from_video(audio_path)
            return extracted_audio, extracted_audio
        return audio_path, None
    if video_path:
        extracted_audio = extract_audio_from_video(video_path)
        return extracted_audio, extracted_audio
    raise ValueError("Загрузите аудио или видеофайл.")


def load_whisper_pipeline():
    global whisper_pipeline, whisper_pipeline_key
    settings = get_model_settings()
    resolved_device = _resolve_whisper_device(settings["whisper_device"])
    compute_type = _resolve_whisper_compute_type(
        settings["whisper_compute_type"],
        resolved_device,
    )
    pipeline_key = (
        settings["whisper_model_name"],
        resolved_device,
        compute_type,
        settings["whisper_cpu_threads"],
        settings["whisper_num_workers"],
    )

    if whisper_pipeline is not None and whisper_pipeline_key == pipeline_key:
        logger.debug("Reusing Whisper pipeline: %s", pipeline_key)
        return whisper_pipeline
    if whisper_pipeline is not None:
        unload_whisper_model()

    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Пакет faster-whisper не установлен. Установите зависимости из requirements.txt."
        ) from exc

    model_kwargs = {
        "device": resolved_device,
        "compute_type": compute_type,
        "cpu_threads": settings["whisper_cpu_threads"],
        "num_workers": settings["whisper_num_workers"],
        "download_root": str(CACHE_DIR / "whisper"),
    }
    if resolved_device == "cuda":
        model_kwargs["device_index"] = 0

    logger.info(
        "Loading Whisper model: model=%s device=%s compute=%s cpu_threads=%s workers=%s",
        settings["whisper_model_name"],
        resolved_device,
        compute_type,
        settings["whisper_cpu_threads"],
        settings["whisper_num_workers"],
    )
    model = WhisperModel(settings["whisper_model_name"], **model_kwargs)
    whisper_pipeline = BatchedInferencePipeline(model=model)
    whisper_pipeline_key = pipeline_key
    logger.info("Whisper model loaded: %s", pipeline_key)
    return whisper_pipeline


def unload_transcription_models():
    global whisper_pipeline, whisper_pipeline_key, diarization_pipeline, diarization_pipeline_name
    if whisper_pipeline is None and diarization_pipeline is None:
        return

    logger.info("Unloading transcription models")
    whisper_pipeline = None
    whisper_pipeline_key = None
    diarization_pipeline = None
    diarization_pipeline_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_segments(audio_path):
    return list(iter_transcribe_segments(audio_path))


def iter_transcribe_segments(audio_path):
    pipeline = load_whisper_pipeline()
    settings = get_model_settings()
    logger.info(
        "Starting Whisper transcription: audio=%s batch=%s",
        audio_path,
        settings["whisper_batch_size"],
    )
    segments, _info = pipeline.transcribe(
        audio_path,
        language="ru",
        beam_size=2,
        best_of=2,
        condition_on_previous_text=True,
        vad_filter=True,
        batch_size=settings["whisper_batch_size"],
        temperature=0.0,
    )
    yield from segments


def _load_pyannote_pipeline(pipeline_cls, model_origin, token=None):
    cache_dir = str(CACHE_DIR / "pyannote")
    kwargs = {"cache_dir": cache_dir}
    if token:
        kwargs["token"] = token

    try:
        logger.info("Loading pyannote pipeline: model=%s local=%s", model_origin, _looks_like_local_path(model_origin))
        return pipeline_cls.from_pretrained(model_origin, **kwargs)
    except TypeError:
        kwargs.pop("token", None)
        if token:
            kwargs["use_auth_token"] = token
        try:
            return pipeline_cls.from_pretrained(model_origin, **kwargs)
        except Exception as exc:
            _raise_pyannote_hub_error(exc)
            raise
    except Exception as exc:
        _raise_pyannote_hub_error(exc)
        raise


def _raise_pyannote_hub_error(exc):
    message = str(exc)
    if "public gated repositories" in message or "gated repo" in message:
        raise RuntimeError(
            "Hugging Face token не может скачать файлы gated-модели pyannote. "
            "Если используется fine-grained token, в настройках токена включите "
            "read-доступ к public gated repositories, к которым у аккаунта уже есть доступ. "
            "Проще: создайте новый token типа Read, сохраните его в интерфейсе и "
            "перезапустите диаризацию. Для pyannote.audio 4.x можно оставить поле "
            "модели пустым и использовать pyannote-community/speaker-diarization-community-1."
        ) from exc


def load_diarization_pipeline(hf_token=None, model_name=None):
    global diarization_pipeline, diarization_pipeline_name
    model_origin, local_model = _resolve_local_model_origin(model_name)

    if diarization_pipeline is not None and diarization_pipeline_name == model_origin:
        logger.debug("Reusing diarization pipeline: %s", model_origin)
        return diarization_pipeline

    if _looks_like_local_path(model_origin) and not local_model:
        raise RuntimeError(f"Локальная модель диаризации не найдена: {model_origin}")

    token = None if local_model else (hf_token or _hf_token())
    if not local_model and not token:
        raise RuntimeError(
            "Для загрузки pyannote из Hugging Face нужен token с доступом к модели. "
            "Либо передайте token в UI/HF_TOKEN, либо укажите локальный путь к "
            "скачанной модели диаризации."
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"\s*torchcodec is not installed correctly.*",
                category=UserWarning,
            )
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Пакет pyannote.audio не установлен. Установите зависимости из requirements.txt."
        ) from exc

    diarization_pipeline = _load_pyannote_pipeline(Pipeline, model_origin, token=token)
    diarization_pipeline_name = model_origin
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))
    logger.info("Diarization pipeline loaded: model=%s cuda=%s", model_origin, torch.cuda.is_available())
    return diarization_pipeline


def diarize_audio(
    audio_path,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    diarization_model_path=None,
):
    logger.info(
        "Running diarization: audio=%s min_speakers=%s max_speakers=%s model=%s",
        audio_path,
        min_speakers,
        max_speakers,
        diarization_model_path or DIARIZATION_MODEL_NAME,
    )
    pipeline = load_diarization_pipeline(
        hf_token=hf_token,
        model_name=diarization_model_path,
    )
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Пакет soundfile не установлен. Он нужен для локальной диаризации."
        ) from exc

    audio_data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_data.T)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    diarization_kwargs = {}
    min_speakers = _optional_int(min_speakers)
    max_speakers = _optional_int(max_speakers)
    if min_speakers:
        diarization_kwargs["min_speakers"] = min_speakers
    if max_speakers:
        diarization_kwargs["max_speakers"] = max_speakers
    result = pipeline(audio_input, **diarization_kwargs)
    logger.info("Diarization completed: audio=%s", audio_path)
    return result


def _diarization_annotation(diarization):
    if hasattr(diarization, "exclusive_speaker_diarization"):
        return diarization.exclusive_speaker_diarization
    if hasattr(diarization, "speaker_diarization"):
        return diarization.speaker_diarization
    return diarization


def diarization_turns(diarization):
    annotation = _diarization_annotation(diarization)
    if not hasattr(annotation, "itertracks"):
        raise RuntimeError(
            f"Неожиданный формат результата диаризации: {type(diarization).__name__}"
        )

    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append((float(turn.start), float(turn.end), str(speaker)))
    return sorted(turns, key=lambda item: (item[0], item[1]))


def find_segment_speaker(segment_start, segment_end, turns):
    scores = defaultdict(float)
    for turn_start, turn_end, speaker in turns:
        overlap = min(segment_end, turn_end) - max(segment_start, turn_start)
        if overlap > 0:
            scores[speaker] += overlap
    if not scores:
        return "SPEAKER_UNKNOWN"
    return max(scores, key=scores.get)


def format_plain_transcript(segments):
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return remove_exact_sentence_repeats(text)


def format_diarized_transcript(segments, turns):
    lines = []
    previous_text = None
    for segment in segments:
        text = segment.text.strip()
        if not text or text == previous_text:
            continue
        start = float(segment.start or 0)
        end = float(segment.end or start)
        speaker = find_segment_speaker(start, end, turns)
        lines.append(
            f"[{_format_timestamp(start)} - {_format_timestamp(end)}] {speaker}: {text}"
        )
        previous_text = text
    return "\n".join(lines)


def transcribe_audio(
    audio_path,
    video_path=None,
    enable_diarization=False,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    diarization_model_path=None,
):
    """Transcribe uploaded audio or a video's audio track, optionally with speaker diarization."""
    temp_audio_path = None
    try:
        logger.info(
            "Transcription requested: audio=%s video=%s diarization=%s",
            bool(audio_path),
            bool(video_path),
            bool(enable_diarization),
        )
        source_audio_path, temp_audio_path = get_audio_source(
            audio_path,
            video_path,
            force_wav=enable_diarization,
        )
        segments = transcribe_segments(source_audio_path)
        logger.info("Transcription segments completed: count=%s source=%s", len(segments), source_audio_path)

        if not enable_diarization:
            transcript = format_plain_transcript(segments)
            logger.info("Plain transcription completed: chars=%s", len(transcript))
            return transcript

        try:
            diarization = diarize_audio(
                source_audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
                diarization_model_path=diarization_model_path,
            )
            turns = diarization_turns(diarization)
            transcript = format_diarized_transcript(segments, turns)
            logger.info(
                "Diarized transcription completed: chars=%s turns=%s",
                len(transcript),
                len(turns),
            )
            return transcript
        except Exception as exc:
            logger.exception("Diarization failed after transcription")
            transcript = format_plain_transcript(segments)
            return f"{transcript}\n\n[Диаризация не выполнена: {exc}]"

    except Exception as exc:
        logger.exception("Transcription failed")
        return f"Ошибка транскрибации: {exc}"
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.debug("Temporary audio removed: %s", temp_audio_path)


def transcribe_audio_live(
    audio_path,
    video_path=None,
    enable_diarization=False,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    diarization_model_path=None,
):
    """Yield progressively updated transcription text for Gradio streaming outputs."""
    temp_audio_path = None
    segments = []
    final_text = ""

    try:
        logger.info(
            "Live transcription requested: audio=%s video=%s diarization=%s",
            bool(audio_path),
            bool(video_path),
            bool(enable_diarization),
        )
        yield "Подготовка аудио..."
        source_audio_path, temp_audio_path = get_audio_source(
            audio_path,
            video_path,
            force_wav=enable_diarization,
        )

        if enable_diarization:
            yield (
                "Идет потоковая транскрибация...\n"
                "Диаризация будет применена к финальному тексту после обработки файла."
            )
        else:
            yield "Идет потоковая транскрибация..."

        last_yield_end = 0.0
        for segment in iter_transcribe_segments(source_audio_path):
            segments.append(segment)
            segment_end = float(segment.end or 0)
            if segment_end - last_yield_end >= LIVE_TRANSCRIPTION_UPDATE_SECONDS:
                final_text = format_plain_transcript(segments)
                logger.debug(
                    "Live transcription partial: segments=%s chars=%s",
                    len(segments),
                    len(final_text),
                )
                yield final_text
                last_yield_end = segment_end

        final_text = format_plain_transcript(segments)
        logger.info("Live transcription completed: segments=%s chars=%s", len(segments), len(final_text))
        if not enable_diarization:
            yield final_text
            return

        yield f"{final_text}\n\n[Применяю диаризацию к полной аудиодорожке...]"
        try:
            diarization = diarize_audio(
                source_audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
                diarization_model_path=diarization_model_path,
            )
            turns = diarization_turns(diarization)
            diarized_text = format_diarized_transcript(segments, turns)
            logger.info(
                "Live diarized transcription completed: chars=%s turns=%s",
                len(diarized_text),
                len(turns),
            )
            yield diarized_text
        except Exception as exc:
            logger.exception("Live diarization failed after transcription")
            yield f"{final_text}\n\n[Диаризация не выполнена: {exc}]"

    except Exception as exc:
        logger.exception("Live transcription failed")
        yield f"Ошибка потоковой транскрибации: {exc}"
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.debug("Temporary audio removed: %s", temp_audio_path)


def load_hf_model(model_name=None):
    global hf_tokenizer, hf_model, hf_model_name
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = (model_name or get_model_settings()["summary_model_name"]).strip()
    if hf_tokenizer is not None and hf_model is not None and hf_model_name == model_name:
        logger.debug("Reusing LLM model: %s", model_name)
        return hf_tokenizer, hf_model

    unload_transcription_models()

    token = _hf_token()
    common_kwargs = {
        "trust_remote_code": True,
        "cache_dir": CACHE_DIR,
    }
    if token:
        common_kwargs["token"] = token

    try:
        logger.info("Loading LLM tokenizer: %s", model_name)
        processor = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
    except Exception as exc:
        logger.exception("LLM tokenizer loading failed: %s", model_name)
        raise RuntimeError(
            f"Не удалось загрузить токенизатор для {model_name}. "
            "Для Gemma 4 нужен transformers>=5.9.0 и huggingface-hub>=1.5.0; "
            "если модель gated, сохраните Hugging Face token в интерфейсе."
        ) from exc

    model_kwargs = {
        **common_kwargs,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "sdpa"

    logger.info(
        "Loading LLM model: model=%s cuda=%s dtype=%s",
        model_name,
        torch.cuda.is_available(),
        model_kwargs["torch_dtype"],
    )
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    hf_tokenizer = processor
    hf_model_name = model_name
    logger.info("LLM model loaded: %s", model_name)
    return hf_tokenizer, hf_model


def _get_tokenizer(processor):
    return getattr(processor, "tokenizer", processor)


def split_text_into_chunks(text, processor, max_tokens=SUMMARY_MAX_CHUNK_SIZE):
    """Split text into chunks that don't exceed max_tokens."""
    tokenizer = _get_tokenizer(processor)
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks


def split_text_into_approx_token_chunks(text, max_tokens=SUMMARY_MAX_CHUNK_SIZE):
    """Split text without loading a tokenizer, for remote OpenAI-compatible backends."""
    max_chars = max(1000, int(max_tokens) * 4)
    paragraphs = re.split(r"(\n\s*\n)", text)
    chunks = []
    current = ""

    for part in paragraphs:
        if not part:
            continue
        if len(part) > max_chars:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            for index in range(0, len(part), max_chars):
                chunk = part[index : index + max_chars].strip()
                if chunk:
                    chunks.append(chunk)
            continue
        if len(current) + len(part) > max_chars and current.strip():
            chunks.append(current.strip())
            current = part
        else:
            current += part

    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def prepare_text_for_summary(text):
    text = re.sub(r"^\[Сегмент \d+\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*",
        "",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_generated_text(text):
    text = (text or "").strip()
    text = re.sub(r"</?\w+[^>]*>", "", text)
    text = re.sub(r"\biclee?\s*(?:\([^)]*\))?\s*>?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bí?culos\b", "", text, flags=re.IGNORECASE)

    pieces = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []
    for piece in pieces:
        stripped = piece.strip()
        if not stripped:
            continue
        cyrillic = len(re.findall(r"[А-Яа-яЁё]", stripped))
        latin = len(re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]", stripped))
        if latin > 40 and cyrillic == 0:
            continue
        cleaned.append(stripped)

    text = " ".join(cleaned)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_user_prompt(template, default_template, summary):
    template = (template or "").strip() or default_template
    if "{summary}" in template:
        return template.replace("{summary}", summary)
    if "{text}" in template:
        return template.replace("{text}", summary)
    return f"{template}\n\nМатериал:\n{summary}"


def build_chat_messages(prompt, system_prompt=None):
    messages = []
    system_prompt = (system_prompt or "").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_chat_inputs(processor, prompt, model, system_prompt=None):
    messages = build_chat_messages(prompt, system_prompt=system_prompt)
    tokenizer = _get_tokenizer(processor)

    try:
        chat_template_kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        try:
            input_tensor = tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **chat_template_kwargs,
            )
        except TypeError:
            input_tensor = tokenizer.apply_chat_template(
                messages,
                **chat_template_kwargs,
            )
        if isinstance(input_tensor, dict):
            input_len = input_tensor["input_ids"].shape[-1]
            input_tensor = {
                key: value.to(model.device) if hasattr(value, "to") else value
                for key, value in input_tensor.items()
            }
            return input_tensor, input_len
        input_tensor = input_tensor.to(model.device)
        return input_tensor, input_tensor.shape[1]
    except Exception:
        if hasattr(tokenizer, "apply_chat_template"):
            chat_template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=False,
                    **chat_template_kwargs,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    **chat_template_kwargs,
                )
        else:
            prompt_text = prompt
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        return inputs, inputs["input_ids"].shape[-1]


def build_generation_config(
    model_name,
    model,
    processor,
    max_new_tokens=NEW_TOKENS,
    generation_settings=None,
):
    from transformers import GenerationConfig

    generation_settings = _normalize_generation_settings(generation_settings)
    generation_kwargs = {"cache_dir": CACHE_DIR}
    token = _hf_token()
    if token:
        generation_kwargs["token"] = token
    try:
        generation_config = GenerationConfig.from_pretrained(
            model_name,
            **generation_kwargs,
        )
    except Exception:
        generation_config = model.generation_config
    generation_config.temperature = generation_settings["llm_temperature"]
    generation_config.top_p = generation_settings["llm_top_p"]
    generation_config.top_k = generation_settings["llm_top_k"]
    generation_config.repetition_penalty = generation_settings["llm_repetition_penalty"]
    generation_config.max_new_tokens = max_new_tokens
    generation_config.no_repeat_ngram_size = generation_settings[
        "llm_no_repeat_ngram_size"
    ]
    base_tokenizer = _get_tokenizer(processor)
    generation_config.eos_token_id = base_tokenizer.eos_token_id
    generation_config.pad_token_id = (
        base_tokenizer.pad_token_id or base_tokenizer.eos_token_id
    )
    generation_config.do_sample = generation_settings["llm_do_sample"]
    logger.debug(
        "Generation config prepared: model=%s max_new_tokens=%s do_sample=%s temperature=%s top_p=%s top_k=%s repetition_penalty=%s no_repeat_ngram_size=%s",
        model_name,
        max_new_tokens,
        generation_config.do_sample,
        generation_config.temperature,
        generation_config.top_p,
        generation_config.top_k,
        generation_config.repetition_penalty,
        generation_config.no_repeat_ngram_size,
    )
    return generation_config


def generate_chat_text(
    processor,
    model,
    prompt,
    generation_config,
    clean_output=True,
    system_prompt=None,
):
    inputs, input_len = build_chat_inputs(
        processor,
        prompt,
        model,
        system_prompt=system_prompt,
    )
    if isinstance(inputs, dict):
        outputs = model.generate(**inputs, generation_config=generation_config)
    else:
        outputs = model.generate(inputs, generation_config=generation_config)
    tokenizer = _get_tokenizer(processor)
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return clean_generated_text(text) if clean_output else text.strip()


def vllm_chat_url(base_url):
    return f"{_normalize_vllm_base_url(base_url)}/chat/completions"


def generate_vllm_chat_text(
    base_url,
    model_name,
    prompt,
    max_new_tokens=NEW_TOKENS,
    clean_output=True,
    system_prompt=None,
    generation_settings=None,
):
    generation_settings = _normalize_generation_settings(generation_settings)
    temperature = (
        generation_settings["llm_temperature"]
        if generation_settings["llm_do_sample"]
        else 0.0
    )
    payload = {
        "model": model_name,
        "messages": build_chat_messages(prompt, system_prompt=system_prompt),
        "temperature": temperature,
        "top_p": generation_settings["llm_top_p"],
        "max_tokens": int(max_new_tokens),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if generation_settings["llm_top_k"] > 0:
        payload["top_k"] = generation_settings["llm_top_k"]
    if generation_settings["llm_repetition_penalty"] != 1.0:
        payload["repetition_penalty"] = generation_settings["llm_repetition_penalty"]
    if generation_settings["llm_no_repeat_ngram_size"] > 0:
        payload["no_repeat_ngram_size"] = generation_settings["llm_no_repeat_ngram_size"]
    headers = {"Content-Type": "application/json"}
    api_key = (os.getenv("VLLM_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def send_request(request_payload):
        request = urllib.request.Request(
            vllm_chat_url(base_url),
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=VLLM_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        logger.debug(
            "Sending vLLM chat request: url=%s model=%s max_tokens=%s",
            vllm_chat_url(base_url),
            model_name,
            max_new_tokens,
        )
        response_data = send_request(payload)
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")[-2000:]
        retry_payload = dict(payload)
        retry_removed_keys = [
            key
            for key in (
                "chat_template_kwargs",
                "top_k",
                "repetition_penalty",
                "no_repeat_ngram_size",
            )
            if key in retry_payload
        ]
        if retry_removed_keys:
            logger.warning(
                "vLLM rejected optional request fields, retrying without %s: %s",
                retry_removed_keys,
                error_body,
            )
            for key in retry_removed_keys:
                retry_payload.pop(key, None)
            try:
                response_data = send_request(retry_payload)
            except urllib.error.HTTPError as retry_exc:
                retry_body = retry_exc.read().decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"vLLM вернул HTTP {retry_exc.code}: {retry_body}"
                ) from retry_exc
        else:
            raise RuntimeError(
                f"vLLM вернул HTTP {exc.code}: {error_body}"
            ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Не удалось подключиться к vLLM по адресу {base_url}: {exc}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"vLLM не ответил за {VLLM_TIMEOUT_SECONDS} секунд по адресу {base_url}"
        ) from exc

    try:
        text = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Неожиданный ответ vLLM: {response_data}") from exc

    return clean_generated_text(text) if clean_output else (text or "").strip()


def build_llm_runtime(settings, model_name=None, max_new_tokens=None):
    backend = _normalize_llm_backend(settings.get("llm_backend"))
    generation_settings = _normalize_generation_settings(settings)
    max_new_tokens = _positive_int(
        max_new_tokens,
        settings["summary_max_new_tokens"],
        minimum=64,
    )

    if backend == LLM_BACKEND_VLLM:
        runtime = {
            "backend": LLM_BACKEND_VLLM,
            "base_url": _normalize_vllm_base_url(settings["vllm_base_url"]),
            "model_name": (model_name or settings["vllm_model_name"]).strip(),
            "max_new_tokens": max_new_tokens,
            "system_prompt": settings.get("llm_system_prompt", ""),
            "generation_settings": generation_settings,
        }
        logger.info(
            "Using vLLM backend: base_url=%s model=%s max_new_tokens=%s do_sample=%s temperature=%s top_p=%s",
            runtime["base_url"],
            runtime["model_name"],
            runtime["max_new_tokens"],
            generation_settings["llm_do_sample"],
            generation_settings["llm_temperature"],
            generation_settings["llm_top_p"],
        )
        unload_hf_model()
        return runtime

    runtime_model_name = (model_name or settings["summary_model_name"]).strip()
    tokenizer, model = load_hf_model(runtime_model_name)
    generation_config = build_generation_config(
        runtime_model_name,
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        generation_settings=generation_settings,
    )
    return {
        "backend": LLM_BACKEND_TRANSFORMERS,
        "model_name": runtime_model_name,
        "tokenizer": tokenizer,
        "model": model,
        "generation_config": generation_config,
        "max_new_tokens": max_new_tokens,
        "system_prompt": settings.get("llm_system_prompt", ""),
        "generation_settings": generation_settings,
    }


def split_text_for_llm_runtime(text, runtime, max_chunk_size):
    if runtime["backend"] == LLM_BACKEND_VLLM:
        return split_text_into_approx_token_chunks(text, max_tokens=max_chunk_size)
    return split_text_into_chunks(
        text,
        runtime["tokenizer"],
        max_tokens=max_chunk_size,
    )


def generate_llm_text(runtime, prompt, clean_output=True):
    if runtime["backend"] == LLM_BACKEND_VLLM:
        return generate_vllm_chat_text(
            runtime["base_url"],
            runtime["model_name"],
            prompt,
            max_new_tokens=runtime["max_new_tokens"],
            clean_output=clean_output,
            system_prompt=runtime.get("system_prompt"),
            generation_settings=runtime.get("generation_settings"),
        )
    return generate_chat_text(
        runtime["tokenizer"],
        runtime["model"],
        prompt,
        runtime["generation_config"],
        clean_output=clean_output,
        system_prompt=runtime.get("system_prompt"),
    )


def render_text_task_prompt(instruction, text):
    instruction = (instruction or "").strip()
    text = (text or "").strip()
    if "{text}" in instruction:
        return instruction.replace("{text}", text)
    return (
        "Выполни пользовательскую задачу по тексту ниже.\n"
        "Правила: опирайся только на данный текст; не выдумывай факты; "
        "если данных недостаточно, прямо укажи это.\n\n"
        f"Задача:\n{instruction}\n\n"
        f"Текст:\n{text}"
    )


def process_text_with_instruction(
    text,
    instruction,
    model_name=None,
    max_chunk_size=None,
    max_new_tokens=None,
):
    """Run a free-form user instruction against pasted text or a transcript."""
    try:
        if not text or not text.strip():
            return "Передайте текст или транскрипцию для обработки."
        if not instruction or not instruction.strip():
            return "Опишите, что нужно сделать с текстом."

        logger.info(
            "Free-form text task requested: input_chars=%s instruction_chars=%s",
            len(text),
            len(instruction),
        )
        text = prepare_text_for_summary(text)
        instruction = instruction.strip()
        settings = get_model_settings()
        max_chunk_size = _positive_int(
            max_chunk_size,
            settings["summary_max_chunk_size"],
            minimum=256,
        )
        runtime = build_llm_runtime(settings, model_name=model_name, max_new_tokens=max_new_tokens)
        chunks = split_text_for_llm_runtime(text, runtime, max_chunk_size)
        logger.info(
            "Free-form text task chunks prepared: chunks=%s max_chunk_size=%s max_new_tokens=%s model=%s backend=%s",
            len(chunks),
            max_chunk_size,
            runtime["max_new_tokens"],
            runtime["model_name"],
            runtime["backend"],
        )

        if len(chunks) == 1:
            result = generate_llm_text(
                runtime,
                render_text_task_prompt(instruction, chunks[0]),
                clean_output=False,
            )
            logger.info("Free-form text task completed: output_chars=%s", len(result))
            return result

        partial_results = []
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            prompt = (
                "Ниже фрагмент длинной транскрипции. Выполни пользовательскую задачу "
                "только по фактам из этого фрагмента. Если задача требует общего вывода "
                "по всей транскрипции, подготовь промежуточные наблюдения для объединения.\n\n"
                f"Задача:\n{instruction}\n\n"
                f"Фрагмент {index} из {total_chunks}:\n{chunk}"
            )
            partial = generate_llm_text(
                runtime,
                prompt,
                clean_output=False,
            )
            partial_results.append(f"[Фрагмент {index}]\n{partial}")

        final_prompt = (
            "Ниже частичные ответы модели по фрагментам одной транскрипции. "
            "Собери единый ответ на исходную пользовательскую задачу: убери повторы, "
            "объедини близкие пункты, сохрани структуру и не добавляй фактов вне частичных ответов.\n\n"
            f"Исходная задача:\n{instruction}\n\n"
            "Частичные ответы:\n"
            + "\n\n".join(partial_results)
        )
        result = generate_llm_text(
            runtime,
            final_prompt,
            clean_output=False,
        )
        logger.info("Free-form text task completed: output_chars=%s partials=%s", len(result), len(partial_results))
        return result

    except Exception as exc:
        logger.exception("Free-form text task failed")
        return f"Ошибка обработки текста: {exc}"


def summarize_text_with_prompt_optimized(
    text,
    model_name=None,
    max_chunk_size=None,
    max_new_tokens=None,
    make_protocol=True,
    summary_prompt=None,
    protocol_prompt=None,
):
    """
    Делает суммаризацию по чанкам. Если make_protocol=True,
    дополнительно на основе общей суммаризации формирует протокол.
    """
    try:
        if not text or not text.strip():
            return "Передайте текст для суммаризации."

        logger.info("Summarization requested: input_chars=%s make_protocol=%s", len(text), make_protocol)
        text = prepare_text_for_summary(text)
        settings = get_model_settings()
        max_chunk_size = _positive_int(
            max_chunk_size,
            settings["summary_max_chunk_size"],
            minimum=256,
        )
        runtime = build_llm_runtime(settings, model_name=model_name, max_new_tokens=max_new_tokens)
        chunks = split_text_for_llm_runtime(text, runtime, max_chunk_size)
        logger.info(
            "Summarization chunks prepared: chunks=%s max_chunk_size=%s max_new_tokens=%s model=%s backend=%s",
            len(chunks),
            max_chunk_size,
            runtime["max_new_tokens"],
            runtime["model_name"],
            runtime["backend"],
        )
        summaries = []

        prompt_template = (
            "Ты делаешь краткую выжимку фрагмента русскоязычной транскрипции.\n"
            "Правила: отвечай только на русском языке; не используй английский, испанский, "
            "HTML/XML-теги и служебные слова; не добавляй факты, которых нет во фрагменте.\n"
            "Верни 5-8 содержательных тезисов без вступления.\n\n"
            "Фрагмент:\n{}"
        )

        for chunk in chunks:
            prompt = prompt_template.format(chunk)
            summary = generate_llm_text(runtime, prompt)
            summaries.append(summary)

        combined_summary = clean_generated_text("\n".join(summaries))

        if not make_protocol:
            final_summary_prompt = render_user_prompt(
                summary_prompt,
                DEFAULT_SUMMARY_PROMPT,
                combined_summary,
            )
            result = generate_llm_text(
                runtime,
                final_summary_prompt,
            )
            logger.info("Summary completed: output_chars=%s chunk_summaries=%s", len(result), len(summaries))
            return result

        final_prompt = render_user_prompt(
            protocol_prompt,
            DEFAULT_PROTOCOL_PROMPT,
            combined_summary,
        )

        try:
            final_summary = generate_llm_text(
                runtime,
                final_prompt,
            )

            if final_summary.startswith(final_prompt):
                final_summary = final_summary[len(final_prompt) :].strip()

        except Exception as exc:
            logger.exception("Final protocol generation failed, returning combined summary")
            final_summary = combined_summary

        logger.info("Protocol completed: output_chars=%s chunk_summaries=%s", len(final_summary), len(summaries))
        return final_summary

    except Exception as exc:
        logger.exception("Summarization failed")
        return f"Ошибка суммаризации: {exc}"
