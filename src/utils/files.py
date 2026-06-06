from __future__ import annotations

import re
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def sanitize_filename(name: str, replacement: str = "_") -> str:
    raw_name = str(name)
    suffix = ""
    stem = raw_name
    if "." in raw_name and not raw_name.endswith("."):
        stem, suffix = raw_name.rsplit(".", 1)
        suffix = f".{suffix.lower()}"
    safe_stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', replacement, stem)
    safe_stem = re.sub(r"\s+", " ", safe_stem).strip(" .")
    safe_stem = re.sub(r"_+", "_", safe_stem)
    if not safe_stem:
        safe_stem = "video"
    return f"{safe_stem}{suffix}"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def safe_output_path(input_file: Path, input_root: Path, output_root: Path, suffix: str = ".mp4") -> Path:
    resolved_input = input_file.resolve()
    resolved_input_root = input_root.resolve()
    resolved_output_root = output_root.resolve()
    relative_parent = resolved_input.parent.relative_to(resolved_input_root)
    output_dir = resolved_output_root / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(resolved_input.with_suffix(suffix).name)
    return unique_path(output_dir / safe_name)


def find_video_files(input_dir: Path, recursive: bool = True) -> list[Path]:
    if not input_dir.exists():
        return []
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def find_media_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def list_lut_files(lut_dir: Path) -> list[Path]:
    if not lut_dir.exists():
        return []
    return sorted(path for path in lut_dir.rglob("*.cube") if path.is_file())
