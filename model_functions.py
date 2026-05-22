import os
import re
import subprocess
import tempfile
import gc
import warnings
from collections import defaultdict
from pathlib import Path

import torch

# Configuration
TEMPERATURE = 0.3
NEW_TOKENS = 800
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME", "google/gemma-4-E4B-it")
SUMMARY_MAX_CHUNK_SIZE = int(os.getenv("SUMMARY_MAX_CHUNK_SIZE", "4096"))
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3")
LIVE_TRANSCRIPTION_UPDATE_SECONDS = float(
    os.getenv("LIVE_TRANSCRIPTION_UPDATE_SECONDS", "5")
)
DIARIZATION_MODEL_NAME = os.getenv(
    "DIARIZATION_MODEL_NAME",
    "pyannote-community/speaker-diarization-community-1",
)
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

# Cache
CACHE_DIR = Path.cwd() / "model_cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)
HF_TOKEN_FILE = Path(os.getenv("HF_TOKEN_FILE", str(CACHE_DIR / "hf_token.txt")))

hf_tokenizer = None
hf_model = None
hf_model_name = None
whisper_pipeline = None
diarization_pipeline = None
diarization_pipeline_name = None


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
        raise RuntimeError(f"Не удалось извлечь аудиодорожку из видео через ffmpeg: {stderr}")

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
    global whisper_pipeline
    if whisper_pipeline is not None:
        return whisper_pipeline

    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Пакет faster-whisper не установлен. Установите зависимости из requirements.txt."
        ) from exc

    cuda_available = torch.cuda.is_available()
    model_kwargs = {
        "device": "cuda" if cuda_available else "cpu",
        "compute_type": "float16" if cuda_available else "int8",
        "cpu_threads": int(os.getenv("WHISPER_CPU_THREADS", "12")),
        "num_workers": int(os.getenv("WHISPER_NUM_WORKERS", "6")),
        "download_root": str(CACHE_DIR / "whisper"),
    }
    if cuda_available:
        model_kwargs["device_index"] = 0

    model = WhisperModel(WHISPER_MODEL_NAME, **model_kwargs)
    whisper_pipeline = BatchedInferencePipeline(model=model)
    return whisper_pipeline


def unload_transcription_models():
    global whisper_pipeline, diarization_pipeline, diarization_pipeline_name
    if whisper_pipeline is None and diarization_pipeline is None:
        return

    whisper_pipeline = None
    diarization_pipeline = None
    diarization_pipeline_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def transcribe_segments(audio_path):
    return list(iter_transcribe_segments(audio_path))


def iter_transcribe_segments(audio_path):
    pipeline = load_whisper_pipeline()
    segments, _info = pipeline.transcribe(
        audio_path,
        language="ru",
        beam_size=2,
        best_of=2,
        condition_on_previous_text=True,
        vad_filter=True,
        batch_size=int(os.getenv("WHISPER_BATCH_SIZE", "32")),
        temperature=0.0,
    )
    yield from segments


def _load_pyannote_pipeline(pipeline_cls, model_origin, token=None):
    cache_dir = str(CACHE_DIR / "pyannote")
    kwargs = {"cache_dir": cache_dir}
    if token:
        kwargs["token"] = token

    try:
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
    return diarization_pipeline


def diarize_audio(
    audio_path,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    diarization_model_path=None,
):
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
    return pipeline(audio_input, **diarization_kwargs)


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
        source_audio_path, temp_audio_path = get_audio_source(
            audio_path,
            video_path,
            force_wav=enable_diarization,
        )
        segments = transcribe_segments(source_audio_path)

        if not enable_diarization:
            return format_plain_transcript(segments)

        try:
            diarization = diarize_audio(
                source_audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                hf_token=hf_token,
                diarization_model_path=diarization_model_path,
            )
            turns = diarization_turns(diarization)
            return format_diarized_transcript(segments, turns)
        except Exception as exc:
            transcript = format_plain_transcript(segments)
            return f"{transcript}\n\n[Диаризация не выполнена: {exc}]"

    except Exception as exc:
        print(f"Transcription error: {exc}")
        return f"Ошибка транскрибации: {exc}"
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


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
                yield final_text
                last_yield_end = segment_end

        final_text = format_plain_transcript(segments)
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
            yield format_diarized_transcript(segments, turns)
        except Exception as exc:
            yield f"{final_text}\n\n[Диаризация не выполнена: {exc}]"

    except Exception as exc:
        print(f"Live transcription error: {exc}")
        yield f"Ошибка потоковой транскрибации: {exc}"
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


def load_hf_model(model_name=SUMMARY_MODEL_NAME):
    global hf_tokenizer, hf_model, hf_model_name
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if hf_tokenizer is not None and hf_model is not None and hf_model_name == model_name:
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
        processor = AutoTokenizer.from_pretrained(model_name, **common_kwargs)
    except Exception as exc:
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

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    hf_tokenizer = processor
    hf_model_name = model_name
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


def build_chat_inputs(processor, prompt, model):
    messages = [{"role": "user", "content": prompt}]
    tokenizer = _get_tokenizer(processor)

    try:
        input_tensor = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
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
        prompt_text = (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            if hasattr(tokenizer, "apply_chat_template")
            else prompt
        )
        inputs = tokenizer(prompt_text, return_tensors="pt")
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        return inputs, inputs["input_ids"].shape[-1]


def generate_chat_text(processor, model, prompt, generation_config):
    inputs, input_len = build_chat_inputs(processor, prompt, model)
    if isinstance(inputs, dict):
        outputs = model.generate(**inputs, generation_config=generation_config)
    else:
        outputs = model.generate(inputs, generation_config=generation_config)
    tokenizer = _get_tokenizer(processor)
    return clean_generated_text(
        tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    )


def summarize_text_with_prompt_optimized(
    text,
    model_name=SUMMARY_MODEL_NAME,
    max_chunk_size=SUMMARY_MAX_CHUNK_SIZE,
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

        text = prepare_text_for_summary(text)

        from transformers import GenerationConfig

        tokenizer, model = load_hf_model(model_name)

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
        generation_config.temperature = TEMPERATURE
        generation_config.top_p = 0.7
        generation_config.repetition_penalty = 1.2
        generation_config.max_new_tokens = NEW_TOKENS
        generation_config.no_repeat_ngram_size = 6
        base_tokenizer = _get_tokenizer(tokenizer)
        generation_config.eos_token_id = base_tokenizer.eos_token_id
        generation_config.pad_token_id = (
            base_tokenizer.pad_token_id or base_tokenizer.eos_token_id
        )
        generation_config.do_sample = False

        chunks = split_text_into_chunks(text, tokenizer, max_tokens=max_chunk_size)
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
            summary = generate_chat_text(tokenizer, model, prompt, generation_config)
            summaries.append(summary)

        combined_summary = clean_generated_text("\n".join(summaries))

        if not make_protocol:
            final_summary_prompt = render_user_prompt(
                summary_prompt,
                DEFAULT_SUMMARY_PROMPT,
                combined_summary,
            )
            return generate_chat_text(
                tokenizer,
                model,
                final_summary_prompt,
                generation_config,
            )

        final_prompt = render_user_prompt(
            protocol_prompt,
            DEFAULT_PROTOCOL_PROMPT,
            combined_summary,
        )

        try:
            final_summary = generate_chat_text(
                tokenizer,
                model,
                final_prompt,
                generation_config,
            )

            if final_summary.startswith(final_prompt):
                final_summary = final_summary[len(final_prompt) :].strip()

        except Exception as exc:
            print(f"Ошибка при финальной суммаризации (протокол): {exc}")
            final_summary = combined_summary

        return final_summary

    except Exception as exc:
        print(f"Summarization error: {exc}")
        return f"Ошибка суммаризации: {exc}"
