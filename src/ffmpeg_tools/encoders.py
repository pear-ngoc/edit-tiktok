from __future__ import annotations

import platform
from shutil import which

from ffmpeg_tools.runner import run_command
from models import EncoderConfig, EncoderSelection


BACKEND_TO_ENCODER = {
    "cpu_h264": "libx264",
    "cpu_h265": "libx265",
    "nvidia_h264": "h264_nvenc",
    "nvidia_h265": "hevc_nvenc",
    "videotoolbox_h264": "h264_videotoolbox",
    "videotoolbox_h265": "hevc_videotoolbox",
}


def detect_available_encoders(ffmpeg_bin: str = "ffmpeg") -> list[str]:
    if which(ffmpeg_bin) is None and not ffmpeg_bin.endswith(("ffmpeg", "ffmpeg.exe")):
        return []
    try:
        result = run_command([ffmpeg_bin, "-hide_banner", "-encoders"], check=True)
    except Exception:
        return []
    return parse_encoder_names(result.stdout + "\n" + result.stderr)


def parse_encoder_names(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and len(parts[0]) >= 6 and parts[0][0] in {"V", "A", "S"}:
            names.append(parts[1])
    return sorted(set(names))


def select_encoder(
    config: EncoderConfig,
    available_encoders: list[str],
    *,
    system: str | None = None,
    machine: str | None = None,
    width: int = 1280,
    height: int = 720,
) -> EncoderSelection:
    system = system or platform.system()
    machine = machine or platform.machine()
    requested_backend = config.backend
    backend = config.backend
    fallback_reason: str | None = None
    if backend == "auto":
        backend = _auto_backend(config.codec, available_encoders, system, machine)

    encoder = BACKEND_TO_ENCODER.get(backend)
    if not encoder or encoder not in available_encoders:
        fallback_reason = f"{backend} không khả dụng trong FFmpeg hiện tại"
        backend = "cpu_h265" if config.codec == "h265" and "libx265" in available_encoders else "cpu_h264"
        encoder = BACKEND_TO_ENCODER[backend]

    if encoder not in available_encoders:
        fallback_reason = fallback_reason or f"{encoder} không khả dụng trong FFmpeg hiện tại"
        encoder = "libx264"
        backend = "cpu_h264"

    args = _encoder_args(backend, encoder, config.preset, width, height)
    return EncoderSelection(
        requested_backend=requested_backend,
        backend=backend,
        codec_name=encoder,
        args=args,
        description=f"{backend} dùng {encoder}",
        fallback_reason=fallback_reason,
    )


def _auto_backend(codec: str, available: list[str], system: str, machine: str) -> str:
    wants_h265 = codec == "h265"
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        vt = "hevc_videotoolbox" if wants_h265 else "h264_videotoolbox"
        if vt in available:
            return "videotoolbox_h265" if wants_h265 else "videotoolbox_h264"

    nvenc = "hevc_nvenc" if wants_h265 else "h264_nvenc"
    if system in {"Windows", "Linux"} and nvenc in available:
        return "nvidia_h265" if wants_h265 else "nvidia_h264"

    if wants_h265 and "libx265" in available:
        return "cpu_h265"
    return "cpu_h264"


def _encoder_args(backend: str, encoder: str, preset: str, width: int, height: int) -> list[str]:
    preset = preset if preset in {"fast", "balanced", "quality"} else "balanced"
    if backend.startswith("cpu"):
        crf_map = {"fast": "23", "balanced": "20", "quality": "18"}
        preset_map = {"fast": "veryfast", "balanced": "medium", "quality": "slow"}
        return ["-c:v", encoder, "-preset", preset_map[preset], "-crf", crf_map[preset]]

    if backend.startswith("nvidia"):
        cq_map = {"fast": "25", "balanced": "21", "quality": "18"}
        preset_map = {"fast": "p1", "balanced": "p4", "quality": "p6"}
        return [
            "-c:v",
            encoder,
            "-preset",
            preset_map[preset],
            "-rc",
            "vbr",
            "-cq",
            cq_map[preset],
            "-b:v",
            "0",
        ]

    bitrate = _bitrate_for_resolution(width, height, preset)
    return ["-c:v", encoder, "-b:v", bitrate, "-allow_sw", "1"]


def _bitrate_for_resolution(width: int, height: int, preset: str) -> str:
    pixels = width * height
    scale = {"fast": 0.8, "balanced": 1.0, "quality": 1.4}[preset]
    if pixels <= 1280 * 720:
        mbps = 5 * scale
    elif pixels <= 1920 * 1080:
        mbps = 9 * scale
    elif pixels <= 2560 * 1440:
        mbps = 16 * scale
    else:
        mbps = 28 * scale
    return f"{int(mbps * 1000)}k"
