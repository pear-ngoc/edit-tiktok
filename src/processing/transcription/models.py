from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TranscriptionWord:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float
    words: list[TranscriptionWord] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    backend: str
    text: str
    language: str | None
    duration: float | None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    words: list[TranscriptionWord] = field(default_factory=list)
