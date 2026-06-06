from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from shutil import which
from pathlib import Path

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


@dataclass(slots=True)
class EncoderRuntimeProbe:
    nvidia_runtime_available: bool
    nvidia_runtime_reason: str | None = None


def detect_available_encoders(ffmpeg_bin: str = "ffmpeg") -> list[str]:
    if which(ffmpeg_bin) is None and not ffmpeg_bin.endswith(("ffmpeg", "ffmpeg.exe")):
        return []
    try:
        result = run_command([ffmpeg_bin, "-hide_banner", "-encoders"], check=True)
    except Exception:
        return []
    return parse_encoder_names(result.stdout + "\n" + result.stderr)


@lru_cache(maxsize=1)
def probe_nvidia_runtime() -> EncoderRuntimeProbe:
    system = platform.system()
    if system not in {"Windows", "Linux"}:
        return EncoderRuntimeProbe(False, f"NVIDIA runtime không hỗ trợ trên {system}")

    device_hint = _nvidia_device_hint_available(system)
    if device_hint is not None:
        return device_hint

    nvidia_smi = which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return EncoderRuntimeProbe(True, None)
        except Exception as exc:
            reason = f"nvidia-smi thất bại: {exc.__class__.__name__}: {exc}"
        else:
            reason = "nvidia-smi không xác nhận được runtime NVIDIA"
    else:
        reason = "nvidia-smi không có trong container"

    library_names = ["libcuda.so.1", "libcuda.so"] if system == "Linux" else ["nvcuda.dll"]
    for library_name in library_names:
        try:
            ctypes.CDLL(library_name)
            return EncoderRuntimeProbe(True, None)
        except OSError as exc:
            reason = f"{library_name} không khả dụng: {exc}"

    return EncoderRuntimeProbe(False, reason or "NVIDIA runtime không khả dụng")


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
    runtime_probe: EncoderRuntimeProbe | None = None,
) -> EncoderSelection:
    system = system or platform.system()
    machine = machine or platform.machine()
    requested_backend = config.backend
    backend = config.backend
    fallback_reason: str | None = None
    runtime_probe = runtime_probe or (
        probe_nvidia_runtime() if requested_backend == "auto" or requested_backend.startswith("nvidia") else None
    )
    if backend == "auto":
        backend, fallback_reason = _auto_backend(
            config.codec,
            available_encoders,
            system,
            machine,
            nvenc_runtime_available=runtime_probe.nvidia_runtime_available if runtime_probe else True,
        )

    encoder = BACKEND_TO_ENCODER.get(backend)
    if backend.startswith("nvidia") and runtime_probe and not runtime_probe.nvidia_runtime_available:
        fallback_reason = runtime_probe.nvidia_runtime_reason or "NVIDIA runtime không khả dụng trong container"
        backend = "cpu_h265" if config.codec == "h265" and "libx265" in available_encoders else "cpu_h264"
        encoder = BACKEND_TO_ENCODER[backend]

    if not encoder or encoder not in available_encoders:
        fallback_reason = fallback_reason or f"{backend} không khả dụng trong FFmpeg hiện tại"
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


def _auto_backend(
    codec: str,
    available: list[str],
    system: str,
    machine: str,
    *,
    nvenc_runtime_available: bool,
) -> tuple[str, str | None]:
    wants_h265 = codec == "h265"
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        vt = "hevc_videotoolbox" if wants_h265 else "h264_videotoolbox"
        if vt in available:
            return "videotoolbox_h265" if wants_h265 else "videotoolbox_h264", None

    nvenc = "hevc_nvenc" if wants_h265 else "h264_nvenc"
    if system in {"Windows", "Linux"} and nvenc in available:
        if nvenc_runtime_available:
            return "nvidia_h265" if wants_h265 else "nvidia_h264", None
        return (
            "cpu_h265" if wants_h265 and "libx265" in available else "cpu_h264",
            "h264_nvenc/HEVC NVENC có trong FFmpeg nhưng NVIDIA runtime không khả dụng",
        )

    if wants_h265 and "libx265" in available:
        return "cpu_h265", None
    return "cpu_h264", None


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


def _nvidia_device_hint_available(system: str) -> EncoderRuntimeProbe | None:
    if system == "Linux":
        device_paths = [
            Path("/dev/nvidiactl"),
            Path("/dev/nvidia0"),
            Path("/dev/nvidia-uvm"),
        ]
        if any(path.exists() for path in device_paths):
            return EncoderRuntimeProbe(True, None)
        if os.path.exists("/proc/driver/nvidia/version"):
            return EncoderRuntimeProbe(True, None)
    elif system == "Windows":
        # Windows containers are not the usual target here, but keep the probe symmetrical.
        if os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip() not in {"", "void", "none"}:
            return EncoderRuntimeProbe(True, None)
    return None
