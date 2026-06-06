from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlatformInfo:
    system: str
    machine: str
    python_version: str
    is_apple_silicon: bool
    ffmpeg_path: str | None
    ffprobe_path: str | None


def get_platform_info() -> PlatformInfo:
    system = platform.system()
    machine = platform.machine()
    return PlatformInfo(
        system=system,
        machine=machine,
        python_version=sys.version.split()[0],
        is_apple_silicon=system == "Darwin" and machine in {"arm64", "aarch64"},
        ffmpeg_path=shutil.which("ffmpeg") or shutil.which("ffmpeg.exe"),
        ffprobe_path=shutil.which("ffprobe") or shutil.which("ffprobe.exe"),
    )


def is_python_supported() -> bool:
    return sys.version_info >= (3, 11)
