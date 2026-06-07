from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from utils.runtime_logging import JobRuntimeContext


class TranscriptionBackend(Protocol):
    def transcribe(
        self,
        media_path: "Path",
        language: str | None,
        job_context: JobRuntimeContext | None,
    ) -> "TranscriptionResult":
        ...
