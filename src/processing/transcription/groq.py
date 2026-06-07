from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ffmpeg_tools.runner import run_command
from models import GroqTranscriptionConfig
from utils.runtime_logging import JobRuntimeContext, job_prefix

from .base import TranscriptionBackend
from .models import TranscriptionResult, TranscriptionSegment, TranscriptionWord

LOGGER = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
_PERMANENT_STATUS_CODES = {400, 401, 403, 404, 422}

_HttpPermanentError: type[Exception] = Exception  # subclass checked via message


class GroqPermanentError(Exception):
    pass

_semaphore_cache: dict[int, Any] = {}


def _get_semaphore(max_workers: int) -> Any:
    pid = os.getpid()
    if pid not in _semaphore_cache:
        import threading

        _semaphore_cache[pid] = threading.Semaphore(max_workers)
    return _semaphore_cache[pid]


def _resolve_api_key(config: GroqTranscriptionConfig) -> str:
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key
    return config.api_key.strip()


def _redact_key(api_key: str) -> str:
    if not api_key:
        return "<not set>"
    if len(api_key) <= 8:
        return "***"
    return api_key[:4] + "***" + api_key[-4:]


@dataclass
class _ChunkResult:
    segments: list[TranscriptionSegment]
    words: list[TranscriptionWord]
    language: str | None


@dataclass
class _GroqVerboseResponse:
    text: str
    language: str | None
    duration: float | None
    segments: list[dict[str, Any]] = field(default_factory=list)
    words: list[dict[str, Any]] = field(default_factory=list)


def _parse_verbose_response(data: dict[str, Any]) -> _GroqVerboseResponse:
    # Groq may return null instead of [] for these fields
    segments = data.get("segments")
    words = data.get("words")
    return _GroqVerboseResponse(
        text=data.get("text", ""),
        language=data.get("language"),
        duration=data.get("duration"),
        segments=segments if segments is not None else [],
        words=words if words is not None else [],
    )


def _resolve_language(raw: _GroqVerboseResponse) -> str | None:
    lang = raw.language
    if lang is None:
        return None
    # Groq returns full names like "English", "Vietnamese"; normalize to codes
    _LANG_MAP = {
        "english": "en",
        "vietnamese": "vi",
        "japanese": "ja",
        "korean": "ko",
        "chinese": "zh",
        "polish": "pl",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "portuguese": "pt",
        "italian": "it",
        "russian": "ru",
        "thai": "th",
        "indonesian": "id",
        "malay": "ms",
    }
    normalized = lang.strip().lower()
    return _LANG_MAP.get(normalized, lang)


def _normalize_response(raw: _GroqVerboseResponse, chunk_offset: float = 0.0) -> _ChunkResult:
    segments: list[TranscriptionSegment] = []
    for seg in raw.segments:
        seg_words: list[TranscriptionWord] = []
        seg_word_list = seg.get("words")
        if seg_word_list is not None:
            for w in seg_word_list:
                word_text = str(w.get("word", "")).replace("\n", " ").strip()
                if not word_text:
                    continue
                seg_words.append(
                    TranscriptionWord(
                        text=word_text,
                        start=float(w.get("start", 0.0)) + chunk_offset,
                        end=float(w.get("end", 0.0)) + chunk_offset,
                    )
                )
        segments.append(
            TranscriptionSegment(
                text=str(seg.get("text", "")).replace("\n", " ").strip(),
                start=float(seg.get("start", 0.0)) + chunk_offset,
                end=float(seg.get("end", 0.0)) + chunk_offset,
                words=seg_words,
            )
        )

    # Groq may not include a top-level `words` array; fall back to segment words
    if raw.words:
        all_words = [
            TranscriptionWord(
                text=str(w.get("word", "")).replace("\n", " ").strip(),
                start=float(w.get("start", 0.0)) + chunk_offset,
                end=float(w.get("end", 0.0)) + chunk_offset,
            )
            for w in raw.words
            if str(w.get("word", "")).replace("\n", " ").strip()
        ]
    else:
        # Extract flat word list from segments (Groq default behavior)
        all_words = []
        for seg in segments:
            all_words.extend(seg.words)

    # Use the first successful chunk's language (normalized), fall back to requested language
    resolved_lang = _resolve_language(raw)

    return _ChunkResult(segments=segments, words=all_words, language=resolved_lang)


def _deduplicate_words(words: list[TranscriptionWord]) -> list[TranscriptionWord]:
    if not words:
        return []
    result: list[TranscriptionWord] = [words[0]]
    for w in words[1:]:
        if w.start > result[-1].end + 0.05:
            result.append(w)
        elif w.text != result[-1].text:
            result[-1] = w
    return result


def _split_audio_into_chunks(
    audio_path: Path,
    chunk_duration_seconds: int,
    chunk_overlap_seconds: float,
    temp_dir: Path,
) -> list[Path]:
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )
    total_duration = float(result.stdout.strip() or "0")
    if total_duration <= chunk_duration_seconds:
        return [audio_path]

    chunks: list[Path] = []
    chunk_index = 0
    current_start = 0.0
    while current_start < total_duration:
        chunk_end = min(current_start + chunk_duration_seconds, total_duration)
        chunk_path = temp_dir / f"chunk_{chunk_index:03d}.wav"
        args = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-i",
            str(audio_path),
            "-ss",
            str(current_start),
            "-t",
            str(chunk_end - current_start),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(chunk_path),
        ]
        run_command(args, debug=False)
        chunks.append(chunk_path)
        current_start += chunk_duration_seconds - chunk_overlap_seconds
        chunk_index += 1

    return chunks


def _extract_audio_for_groq(
    video_path: Path,
    output_path: Path,
    config: GroqTranscriptionConfig,
    debug: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        str(config.audio_channels),
        "-ar",
        str(config.audio_sample_rate),
        "-c:a",
        "libmp3lame",
        "-b:a",
        config.audio_bitrate,
        str(output_path),
    ]
    run_command(args, debug=debug)
    return output_path


def _upload_and_transcribe_chunk(
    audio_chunk_path: Path,
    config: GroqTranscriptionConfig,
    language: str | None,
    api_key: str,
    chunk_index: int,
    total_chunks: int,
    job_context: JobRuntimeContext | None,
) -> _ChunkResult:
    import httpx

    base_url = config.base_url.rstrip("/")
    endpoint = f"{base_url}/audio/transcriptions"
    parsed = urlparse(endpoint)
    host = parsed.netloc or "api.groq.com"

    timeout = httpx.Timeout(config.timeout_seconds, connect=config.connect_timeout_seconds)
    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
    }
    files: dict[str, Any] = {
        "file": (
            audio_chunk_path.name.replace(".wav", ".mp3"),
            audio_chunk_path.read_bytes(),
            "audio/mpeg",
        ),
        "model": (None, config.model),
        "temperature": (None, str(config.temperature)),
        "response_format": (None, config.response_format),
    }
    for granularity in config.timestamp_granularities:
        files["timestamp_granularities[]"] = (None, granularity)
    if language and language != "auto":
        files["language"] = (None, language)

    LOGGER.debug(
        "%s [TRANSCRIBE] Groq chunk upload | host=%s | model=%s | audio_size=%s | chunk=%s/%s",
        job_prefix(job_context) if job_context else "[JOB -]",
        host,
        config.model,
        audio_chunk_path.stat().st_size,
        chunk_index + 1,
        total_chunks,
    )

    attempt = 0
    last_exc: Exception | None = None
    retry_delay = config.retry_delay_seconds

    while attempt <= config.retry_attempts:
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                start = time.perf_counter()
                response = client.post(
                    endpoint,
                    headers=headers,
                    files=files,
                )
                elapsed = time.perf_counter() - start

                LOGGER.debug(
                    "%s [TRANSCRIBE] Groq response | status=%s | elapsed=%.2fs | chunk=%s/%s",
                    job_prefix(job_context) if job_context else "[JOB -]",
                    response.status_code,
                    elapsed,
                    chunk_index + 1,
                    total_chunks,
                )

                if response.status_code == 401:
                    raise GroqPermanentError("GROQ API key không hợp lệ (HTTP 401)")

                if response.status_code in _PERMANENT_STATUS_CODES:
                    try:
                        err_data = response.json()
                        err_msg = err_data.get("error", {}).get("message", response.text)
                    except Exception:
                        err_msg = response.text
                    raise GroqPermanentError(f"Lỗi GROQ không thể phục hồi: {err_msg}")

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        wait = min(float(retry_after), 60.0)
                    else:
                        wait = retry_delay
                    LOGGER.warning(
                        "%s [TRANSCRIBE] Groq rate limit (429) | chunk=%s/%s | wait=%.1fs",
                        job_prefix(job_context) if job_context else "[JOB -]",
                        chunk_index + 1,
                        total_chunks,
                        wait,
                    )
                    time.sleep(wait)
                    retry_delay = min(retry_delay * 2, 60.0)
                    attempt += 1
                    continue

                if response.status_code in _TRANSIENT_STATUS_CODES:
                    LOGGER.warning(
                        "%s [TRANSCRIBE] Groq transient error %s | chunk=%s/%s | retry=%s/%s",
                        job_prefix(job_context) if job_context else "[JOB -]",
                        response.status_code,
                        chunk_index + 1,
                        total_chunks,
                        attempt + 1,
                        config.retry_attempts,
                    )
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60.0)
                    attempt += 1
                    continue

                response.raise_for_status()
                data = response.json()
                raw = _parse_verbose_response(data)
                return _normalize_response(raw)

        except httpx.ConnectError as exc:
            last_exc = exc
            LOGGER.warning(
                "%s [TRANSCRIBE] Groq connection error | chunk=%s/%s | attempt=%s/%s | %s",
                job_prefix(job_context) if job_context else "[JOB -]",
                chunk_index + 1,
                total_chunks,
                attempt + 1,
                config.retry_attempts,
                exc,
            )
        except httpx.ReadTimeout as exc:
            last_exc = exc
            LOGGER.warning(
                "%s [TRANSCRIBE] Groq read timeout | chunk=%s/%s | attempt=%s/%s",
                job_prefix(job_context) if job_context else "[JOB -]",
                chunk_index + 1,
                total_chunks,
                attempt + 1,
                config.retry_attempts,
            )
        except httpx.TimeoutException as exc:
            last_exc = exc
            LOGGER.warning(
                "%s [TRANSCRIBE] Groq timeout | chunk=%s/%s | attempt=%s/%s",
                job_prefix(job_context) if job_context else "[JOB -]",
                chunk_index + 1,
                total_chunks,
                attempt + 1,
                config.retry_attempts,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in _PERMANENT_STATUS_CODES:
                raise GroqPermanentError(f"Lỗi GROQ không thể phục hồi (HTTP {status}): {exc}")
            last_exc = exc
        except Exception as exc:
            if isinstance(exc, GroqPermanentError):
                raise
            last_exc = exc

        if attempt < config.retry_attempts:
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60.0)
        attempt += 1

    raise RuntimeError(
        f"GROQ transcription thất bại sau {config.retry_attempts + 1} lần thử: {last_exc}"
    )


class GroqTranscriptionBackend:
    def __init__(self, config: GroqTranscriptionConfig, job_context: JobRuntimeContext | None = None) -> None:
        self.config = config
        self.job_context = job_context

    def transcribe(
        self,
        media_path: Path,
        language: str | None,
        job_context: JobRuntimeContext | None,
    ) -> TranscriptionResult:
        resolved_context = job_context or self.job_context
        ctx = resolved_context
        api_key = _resolve_api_key(self.config)
        if not api_key:
            raise RuntimeError("GROQ_API_KEY không được cấu hình")

        redacted = _redact_key(api_key)
        LOGGER.info(
            "%s [TRANSCRIBE] Groq transcription start | media=%s | language=%s | model=%s | api_key=%s",
            job_prefix(ctx) if ctx else "[JOB -]",
            media_path,
            language or "auto",
            self.config.model,
            redacted,
        )

        temp_audio_dir = media_path.parent / "groq_audio_temp"
        temp_audio_dir.mkdir(parents=True, exist_ok=True)
        temp_audio_path = temp_audio_dir / f"groq_{abs(hash(str(media_path)))}.mp3"
        try:
            _extract_audio_for_groq(media_path, temp_audio_path, self.config)
            audio_size = temp_audio_path.stat().st_size

            LOGGER.info(
                "%s [TRANSCRIBE] Groq audio ready | size=%s (%.1fMB) | model=%s",
                job_prefix(ctx) if ctx else "[JOB -]",
                audio_size,
                audio_size / (1024 * 1024),
                self.config.model,
            )

            chunks: list[Path]
            if self.config.chunking_enabled:
                chunks = _split_audio_into_chunks(
                    temp_audio_path,
                    self.config.chunk_duration_seconds,
                    self.config.chunk_overlap_seconds,
                    temp_audio_dir,
                )
            else:
                chunks = [temp_audio_path]

            total_chunks = len(chunks)
            if total_chunks > 1:
                LOGGER.info(
                    "%s [TRANSCRIBE] Groq chunks: %s",
                    job_prefix(ctx) if ctx else "[JOB -]",
                    total_chunks,
                )

            semaphore = _get_semaphore(self.config.max_concurrent_requests)
            all_segments: list[TranscriptionSegment] = []
            all_words: list[TranscriptionWord] = []
            chunk_offset = 0.0
            successful_chunks = 0

            first_language: str | None = None
            for i, chunk_path in enumerate(chunks):
                chunk_result = _upload_and_transcribe_chunk(
                    chunk_path,
                    self.config,
                    language,
                    api_key,
                    i,
                    total_chunks,
                    ctx,
                )
                if i == 0 and chunk_result.language:
                    first_language = chunk_result.language

                for seg in chunk_result.segments:
                    seg.start += chunk_offset
                    seg.end += chunk_offset
                    for w in seg.words:
                        w.start += chunk_offset
                        w.end += chunk_offset
                    all_segments.append(seg)
                for w in chunk_result.words:
                    w.start += chunk_offset
                    w.end += chunk_offset
                    all_words.append(w)

                if total_chunks > 1:
                    LOGGER.info(
                        "%s [TRANSCRIBE] Chunk %s/%s completed",
                        job_prefix(ctx) if ctx else "[JOB -]",
                        i + 1,
                        total_chunks,
                    )

                if i < len(chunks) - 1:
                    chunk_offset = all_words[-1].end if all_words else chunk_offset

                successful_chunks += 1

            all_words = _deduplicate_words(all_words)

            if total_chunks > 1:
                LOGGER.info(
                    "%s [TRANSCRIBE] Merge completed | words=%s | segments=%s",
                    job_prefix(ctx) if ctx else "[JOB -]",
                    len(all_words),
                    len(all_segments),
                )

            full_text = " ".join(seg.text for seg in all_segments)
            detected_language = first_language

            LOGGER.info(
                "%s [TRANSCRIBE] Groq completed | chunks=%s | segments=%s | words=%s | language=%s | elapsed=%.2fs",
                job_prefix(ctx) if ctx else "[JOB -]",
                successful_chunks,
                len(all_segments),
                len(all_words),
                detected_language or "auto",
                0.0,
            )

            return TranscriptionResult(
                backend="groq",
                text=full_text,
                language=detected_language,
                duration=all_segments[-1].end if all_segments else None,
                segments=all_segments,
                words=all_words,
            )

        finally:
            import shutil

            if temp_audio_dir.exists():
                shutil.rmtree(temp_audio_dir, ignore_errors=True)
