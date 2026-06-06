from __future__ import annotations

import random
from pathlib import Path

from models import AudioConfig
from utils.files import AUDIO_EXTENSIONS, find_media_files


def choose_optional_audio_assets(project_root: Path, audio: AudioConfig) -> tuple[Path | None, Path | None]:
    ambient = _choose_one(project_root / audio.ambient_dir) if audio.ambient_enabled else None
    bgm = _choose_one(project_root / audio.bgm_dir) if audio.bgm_enabled else None
    return ambient, bgm


def random_eq_values(audio: AudioConfig) -> tuple[float, float]:
    bass = random.uniform(float(audio.eq_bass_range[0]), float(audio.eq_bass_range[1]))
    treble = random.uniform(float(audio.eq_treble_range[0]), float(audio.eq_treble_range[1]))
    return round(bass, 2), round(treble, 2)


def _choose_one(directory: Path) -> Path | None:
    files = find_media_files(directory, AUDIO_EXTENSIONS)
    return random.choice(files) if files else None
