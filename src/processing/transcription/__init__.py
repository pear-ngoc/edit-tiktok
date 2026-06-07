from __future__ import annotations

from .base import TranscriptionBackend
from .groq import GroqTranscriptionBackend
from .local_whisper import LocalWhisperBackend
from .manager import TranscriptionManager
from .models import TranscriptionResult, TranscriptionSegment, TranscriptionWord

__all__ = [
    "TranscriptionBackend",
    "TranscriptionManager",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionWord",
    "GroqTranscriptionBackend",
    "LocalWhisperBackend",
]
