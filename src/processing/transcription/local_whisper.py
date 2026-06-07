from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ffmpeg_tools.runner import run_command
from utils.runtime_logging import JobRuntimeContext, job_prefix, resolve_whisper_runtime

from .base import TranscriptionBackend
from .models import TranscriptionResult, TranscriptionSegment, TranscriptionWord

LOGGER = logging.getLogger(__name__)

_PREFERRED_PUNCTUATION = (".", "?", "!", ",", ";", ":")


def _extract_audio_track(video_path: Path, audio_path: Path, *, debug: bool = False) -> Path:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    run_command(args, debug=debug)
    return audio_path


def _load_whisper_model(config: Any) -> Any:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - import fallback
        raise RuntimeError(
            "faster-whisper chưa được cài đặt. Hãy cài phụ thuộc để bật phụ đề."
        ) from exc

    resolved_runtime = resolve_whisper_runtime(config)
    requested_device = resolved_runtime.resolved_device
    requested_compute = resolved_runtime.resolved_compute_type
    model_name = config.model_size or "medium"
    try:
        LOGGER.info(
            "Đang tải model faster-whisper: %s | device=%s | compute_type=%s",
            model_name,
            requested_device,
            requested_compute,
        )
        return WhisperModel(
            model_name,
            device=requested_device,
            compute_type=requested_compute,
        )
    except Exception as exc:
        if requested_device == "cpu":
            raise
        LOGGER.warning("Không tải được device %s, fallback sang cpu: %s", requested_device, exc)
        return WhisperModel(model_name, device="cpu", compute_type="int8")


def _flatten_words(segments: list[Any]) -> list[TranscriptionWord]:
    words: list[TranscriptionWord] = []
    for segment in segments:
        segment_words = getattr(segment, "words", None) or []
        for word in segment_words:
            text = _clean_whisper_token(getattr(word, "word", "") or getattr(word, "text", ""))
            if not text:
                continue
            start = float(getattr(word, "start", getattr(segment, "start", 0.0)) or 0.0)
            end = float(getattr(word, "end", getattr(segment, "end", start)) or start)
            if end < start:
                end = start
            words.append(TranscriptionWord(text=text, start=start, end=end))
    return words


def _clean_whisper_token(text: str) -> str:
    return str(text).replace("\n", " ").strip()


class LocalWhisperBackend:
    def __init__(self, config: Any) -> None:
        self.config = config
        self._runtime = resolve_whisper_runtime(config)

    def transcribe(
        self,
        media_path: Path,
        language: str | None,
        job_context: JobRuntimeContext | None,
    ) -> TranscriptionResult:
        temp_audio_path = media_path
        temp_audio_created = False

        try:
            probe = None
            try:
                from ffmpeg_tools.probe import probe_video

                probe = probe_video(media_path)
            except Exception:
                pass

            is_audio_file = False
            if probe:
                is_audio_file = not probe.has_audio and probe.duration and probe.duration > 0

            if not is_audio_file:
                from pathlib import Path as P

                temp_dir = media_path.parent / "subtitles_temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                stem = media_path.stem
                temp_audio_path = temp_dir / f"{stem}_{abs(hash(str(media_path)))}.wav"
                _extract_audio_track(media_path, temp_audio_path)
                temp_audio_created = True

            start = time.perf_counter()
            whisper_model = _load_whisper_model(self.config)
            lang = None if language == "auto" else language

            LOGGER.info(
                "%s [TRANSCRIBE] Transcription start | media=%s | language=%s | backend=faster-whisper",
                job_prefix(job_context) if job_context else "[JOB -]",
                media_path,
                lang or "auto",
            )

            segments, info = whisper_model.transcribe(
                str(temp_audio_path),
                language=lang,
                word_timestamps=True,
                vad_filter=True,
                beam_size=5,
            )
            raw_segments = list(segments)
            words = _flatten_words(raw_segments)
            detected_language = getattr(info, "language", None)

            elapsed = time.perf_counter() - start
            LOGGER.info(
                "%s [TRANSCRIBE] Transcription completed in %.2fs | raw_segments=%s | word_timestamps=%s | detected=%s | backend=faster-whisper",
                job_prefix(job_context) if job_context else "[JOB -]",
                elapsed,
                len(raw_segments),
                len(words),
                detected_language or "unknown",
            )

            result_segments: list[TranscriptionSegment] = []
            for seg in raw_segments:
                seg_words = [
                    TranscriptionWord(
                        text=_clean_whisper_token(getattr(w, "word", "") or getattr(w, "text", "")),
                        start=float(getattr(w, "start", getattr(seg, "start", 0.0)) or 0.0),
                        end=float(getattr(w, "end", getattr(seg, "end", 0.0)) or 0.0),
                    )
                    for w in (getattr(seg, "words", None) or [])
                    if (_clean_whisper_token(getattr(w, "word", "") or getattr(w, "text", "")))
                ]
                result_segments.append(
                    TranscriptionSegment(
                        text=_clean_whisper_token(getattr(seg, "text", "") or ""),
                        start=float(getattr(seg, "start", 0.0) or 0.0),
                        end=float(getattr(seg, "end", 0.0) or 0.0),
                        words=seg_words,
                    )
                )

            full_text = " ".join(
                _clean_whisper_token(getattr(seg, "text", "") or "") for seg in raw_segments
            )

            return TranscriptionResult(
                backend="faster-whisper",
                text=full_text,
                language=detected_language,
                duration=getattr(info, "duration", None),
                segments=result_segments,
                words=words,
            )

        finally:
            if temp_audio_created and temp_audio_path.exists():
                temp_audio_path.unlink(missing_ok=True)
