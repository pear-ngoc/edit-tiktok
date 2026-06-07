from __future__ import annotations

import logging
import os
from pathlib import Path

from models import SubtitlesConfig
from utils.runtime_logging import JobRuntimeContext

from .base import TranscriptionBackend
from .groq import GroqTranscriptionBackend
from .local_whisper import LocalWhisperBackend
from .models import TranscriptionResult

LOGGER = logging.getLogger(__name__)

_ALLOWED_BACKENDS = {"auto", "faster-whisper", "groq"}


def _has_groq_credentials(config: SubtitlesConfig) -> bool:
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return True
    return bool(config.groq.api_key.strip())


class TranscriptionManager:
    def __init__(
        self,
        config: SubtitlesConfig,
        job_context: JobRuntimeContext | None = None,
    ) -> None:
        self.config = config
        self.job_context = job_context

    def resolve_backend(self) -> str:
        requested = self.config.backend.strip().lower()
        if requested not in _ALLOWED_BACKENDS:
            LOGGER.warning(
                "Backend phụ đề không hợp lệ: %s. Fallback về auto.",
                requested,
            )
            return self._resolve_auto()

        if requested == "faster-whisper":
            LOGGER.info("[TRANSCRIBE] Backend requested: faster-whisper")
            return "faster-whisper"

        if requested == "auto":
            return self._resolve_auto()

        if requested == "groq":
            if not _has_groq_credentials(self.config):
                LOGGER.warning(
                    "[TRANSCRIBE] Backend groq được yêu cầu nhưng không có GROQ_API_KEY. "
                    "Fallback sang faster-whisper."
                )
                return "faster-whisper"
            LOGGER.info("[TRANSCRIBE] Backend resolved: groq")
            return "groq"

        return self._resolve_auto()

    def _resolve_auto(self) -> str:
        if _has_groq_credentials(self.config):
            LOGGER.info("[TRANSCRIBE] Backend auto resolved: groq")
            return "groq"
        LOGGER.info("[TRANSCRIBE] Backend auto resolved: faster-whisper")
        return "faster-whisper"

    def transcribe(self, media_path: Path, language: str | None) -> TranscriptionResult:
        resolved = self.resolve_backend()
        LOGGER.info(
            "[TRANSCRIBE] Backend resolved: %s",
            resolved,
        )

        if resolved == "groq":
            backend: TranscriptionBackend = GroqTranscriptionBackend(
                self.config.groq,
                self.job_context,
            )
        else:
            backend = LocalWhisperBackend(self.config)

        try:
            return backend.transcribe(media_path, language, self.job_context)
        except Exception as exc:
            if resolved == "groq" and self.config.groq.fallback_to_local:
                LOGGER.warning(
                    "[TRANSCRIBE] GROQ transcription failed: %s. Falling back to local Faster-Whisper.",
                    exc,
                )
                local_backend = LocalWhisperBackend(self.config)
                return local_backend.transcribe(media_path, language, self.job_context)
            raise
