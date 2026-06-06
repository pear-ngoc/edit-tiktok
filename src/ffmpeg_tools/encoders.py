from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from shutil import which
from typing import Iterable

from models import EncoderConfig, EncoderSelection

BACKEND_TO_ENCODER = {
    "cpu_h264": "libx264",
    "cpu_h265": "libx265",
    "nvidia_h264": "h264_nvenc",
    "nvidia_h265": "hevc_nvenc",
    "amd_amf_h264": "h264_amf",
    "amd_amf_h265": "hevc_amf",
    "vaapi_h264": "h264_vaapi",
    "vaapi_h265": "hevc_vaapi",
    "videotoolbox_h264": "h264_videotoolbox",
    "videotoolbox_h265": "hevc_videotoolbox",
}

HARDWARE_FAILURE_PATTERNS = (
    "cannot load libcuda.so.1",
    "no capable devices found",
    "cannot init cuda",
    "driver does not support required nvenc api",
    "failed to initialize nvenc",
    "cannot load amfrt",
    "amf",
    "vaapi",
    "no device available",
    "device creation failed",
    "cannot open device",
    "videotoolbox",
)

_UNAVAILABLE_BACKENDS: dict[str, str] = {}


@dataclass(slots=True)
class EncoderRuntimeProbe:
    nvidia_runtime_available: bool
    nvidia_runtime_reason: str | None = None


@dataclass(slots=True)
class EncoderCapability:
    backend: str
    ffmpeg_encoder: str
    compiled: bool
    runtime_available: bool
    smoke_test_passed: bool
    failure_reason: str | None = None
    device_path: str | None = None
    platform: str = ""
    containerized: bool = False


def detect_available_encoders(ffmpeg_bin: str = "ffmpeg") -> list[str]:
    if which(ffmpeg_bin) is None and not ffmpeg_bin.endswith(("ffmpeg", "ffmpeg.exe")):
        return []
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []
    return parse_encoder_names((result.stdout or "") + "\n" + (result.stderr or ""))


def parse_encoder_names(output: str) -> list[str]:
    names: list[str] = []
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and len(parts[0]) >= 6 and parts[0][0] in {"V", "A", "S"}:
            names.append(parts[1])
    return sorted(set(names))


def probe_nvidia_runtime() -> EncoderRuntimeProbe:
    system = platform.system()
    if system not in {"Windows", "Linux"}:
        return EncoderRuntimeProbe(False, f"NVIDIA runtime không hỗ trợ trên {system}")

    if _nvidia_device_hint_available(system):
        return EncoderRuntimeProbe(True, None)

    nvidia_smi = which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "-L"],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return EncoderRuntimeProbe(True, None)
        except Exception as exc:
            reason = f"nvidia-smi thất bại: {exc.__class__.__name__}: {exc}"
        else:
            reason = "nvidia-smi không xác nhận được runtime NVIDIA"
    else:
        reason = "nvidia-smi không có trong môi trường runtime"

    library_names = ["libcuda.so.1", "libcuda.so"] if system == "Linux" else ["nvcuda.dll"]
    for library_name in library_names:
        try:
            ctypes.CDLL(library_name)
            return EncoderRuntimeProbe(True, None)
        except OSError as exc:
            reason = f"{library_name} không khả dụng: {exc}"

    return EncoderRuntimeProbe(False, reason or "NVIDIA runtime không khả dụng")


@lru_cache(maxsize=32)
def probe_encoder_capabilities(
    available_encoders: tuple[str, ...],
    *,
    ffmpeg_bin: str = "ffmpeg",
    system: str | None = None,
    machine: str | None = None,
    vaapi_device: str | None = None,
    smoke_test_on_startup: bool = True,
) -> dict[str, EncoderCapability]:
    system = system or platform.system()
    machine = machine or platform.machine()
    containerized = _is_containerized()
    normalized = set(available_encoders)
    return {
        backend: _probe_single_backend(
            backend,
            normalized,
            ffmpeg_bin=ffmpeg_bin,
            system=system,
            machine=machine,
            containerized=containerized,
            vaapi_device=vaapi_device,
            smoke_test_on_startup=smoke_test_on_startup,
        )
        for backend in BACKEND_TO_ENCODER
    }


def select_encoder(
    config: EncoderConfig,
    available_encoders: list[str],
    *,
    system: str | None = None,
    machine: str | None = None,
    width: int = 1280,
    height: int = 720,
    ffmpeg_bin: str = "ffmpeg",
    allow_cpu_fallback: bool = True,
    smoke_test_on_startup: bool = True,
    cache_capability_results: bool = True,
    container_gpu_mode: str = "auto",
    vaapi_device: str | None = None,
) -> EncoderSelection:
    system = system or platform.system()
    machine = machine or platform.machine()
    requested_backend = config.backend
    available_tuple = tuple(sorted(set(available_encoders)))
    probe_fn = probe_encoder_capabilities
    if not cache_capability_results:
        probe_fn = probe_encoder_capabilities.__wrapped__  # type: ignore[attr-defined]
    capabilities = probe_fn(
        available_tuple,
        ffmpeg_bin=ffmpeg_bin,
        system=system,
        machine=machine,
        vaapi_device=vaapi_device,
        smoke_test_on_startup=smoke_test_on_startup,
    )

    requested_backend = (requested_backend or "auto").strip().lower()
    fallback_reason: str | None = None
    if requested_backend == "auto":
        selected_backend, fallback_reason = _select_auto_backend(
            config.codec,
            system=system,
            machine=machine,
            capabilities=capabilities,
            container_gpu_mode=container_gpu_mode,
        )
    else:
        selected_backend = requested_backend

    if selected_backend in _UNAVAILABLE_BACKENDS:
        fallback_reason = fallback_reason or _UNAVAILABLE_BACKENDS[selected_backend]
        if allow_cpu_fallback:
            selected_backend = _cpu_backend_for_codec(config.codec)
        else:
            raise RuntimeError(_UNAVAILABLE_BACKENDS[selected_backend])

    selected_capability = capabilities.get(selected_backend)
    if selected_capability is None:
        selected_backend = _cpu_backend_for_codec(config.codec)
        selected_capability = capabilities[selected_backend]
        fallback_reason = fallback_reason or f"Backend không hỗ trợ: {requested_backend}"

    if not selected_capability.smoke_test_passed:
        if selected_backend.startswith(("nvidia", "amd_amf", "vaapi", "videotoolbox")):
            fallback_reason = selected_capability.failure_reason or f"{selected_capability.ffmpeg_encoder} không chạy được ở runtime"
            if allow_cpu_fallback:
                selected_backend = _cpu_backend_for_codec(config.codec)
                selected_capability = capabilities[selected_backend]
            else:
                raise RuntimeError(fallback_reason)
        elif not selected_capability.compiled:
            fallback_reason = selected_capability.failure_reason or f"{selected_capability.ffmpeg_encoder} không có trong FFmpeg"
            if allow_cpu_fallback:
                selected_backend = _cpu_backend_for_codec(config.codec)
                selected_capability = capabilities[selected_backend]
            else:
                raise RuntimeError(fallback_reason)

    if not selected_capability.compiled and allow_cpu_fallback:
        fallback_reason = fallback_reason or f"{selected_capability.ffmpeg_encoder} không có trong FFmpeg"
        selected_backend = _cpu_backend_for_codec(config.codec)
        selected_capability = capabilities[selected_backend]

    args = _encoder_args(selected_backend, selected_capability.ffmpeg_encoder, config.preset, width, height)
    return EncoderSelection(
        requested_backend=requested_backend,
        backend=selected_backend,
        codec_name=selected_capability.ffmpeg_encoder,
        args=args,
        description=f"{selected_backend} dùng {selected_capability.ffmpeg_encoder}",
        fallback_reason=fallback_reason or selected_capability.failure_reason,
    )


def mark_backend_unavailable(backend: str, reason: str) -> None:
    if backend.startswith("cpu"):
        return
    _UNAVAILABLE_BACKENDS[backend] = reason


def backend_is_unavailable(backend: str) -> bool:
    return backend in _UNAVAILABLE_BACKENDS


def unavailable_backend_reason(backend: str) -> str | None:
    return _UNAVAILABLE_BACKENDS.get(backend)


def probe_encoder_runtime_summary(
    available_encoders: list[str],
    *,
    ffmpeg_bin: str = "ffmpeg",
    system: str | None = None,
    machine: str | None = None,
    vaapi_device: str | None = None,
) -> dict[str, EncoderCapability]:
    return probe_encoder_capabilities(
        tuple(sorted(set(available_encoders))),
        ffmpeg_bin=ffmpeg_bin,
        system=system,
        machine=machine,
        vaapi_device=vaapi_device,
        smoke_test_on_startup=True,
    )


def backend_smoke_test(
    ffmpeg_encoder: str,
    *,
    ffmpeg_bin: str = "ffmpeg",
    backend: str = "",
    vaapi_device: str | None = None,
) -> tuple[bool, str | None]:
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=128x128:rate=10:color=black",
        "-t",
        "0.5",
    ]
    if backend.startswith("vaapi"):
        device = vaapi_device or _find_vaapi_device()
        if not device:
            return False, "Không tìm thấy thiết bị VAAPI"
        command.extend(["-vaapi_device", device, "-vf", "format=nv12,hwupload"])
    command.extend(["-c:v", ffmpeg_encoder, "-f", "null", "-"])
    return _run_ffmpeg_probe(command)


def is_hardware_runtime_failure(stderr: str) -> bool:
    text = (stderr or "").lower()
    return any(pattern in text for pattern in HARDWARE_FAILURE_PATTERNS)


def extract_hardware_failure_reason(stderr: str) -> str:
    text = (stderr or "").strip()
    if not text:
        return "FFmpeg hardware encoder failed"
    for line in reversed(text.splitlines()):
        cleaned = line.strip()
        if cleaned:
            return cleaned[:240]
    return text[:240]


def _probe_single_backend(
    backend: str,
    available_encoders: set[str],
    *,
    ffmpeg_bin: str,
    system: str,
    machine: str,
    containerized: bool,
    vaapi_device: str | None,
    smoke_test_on_startup: bool,
) -> EncoderCapability:
    encoder = BACKEND_TO_ENCODER[backend]
    compiled = encoder in available_encoders
    runtime_available, runtime_reason, device_path = _probe_runtime(backend, system, vaapi_device)
    smoke_passed = False
    failure_reason: str | None = None
    if compiled and runtime_available:
        if smoke_test_on_startup:
            smoke_passed, failure_reason = backend_smoke_test(
                encoder,
                ffmpeg_bin=ffmpeg_bin,
                backend=backend,
                vaapi_device=device_path,
            )
        else:
            smoke_passed = True
    else:
        failure_reason = runtime_reason or f"{encoder} không khả dụng"
    if compiled and runtime_available and not smoke_passed and not failure_reason:
        failure_reason = f"{encoder} smoke test thất bại"
    return EncoderCapability(
        backend=backend,
        ffmpeg_encoder=encoder,
        compiled=compiled,
        runtime_available=runtime_available,
        smoke_test_passed=smoke_passed or (compiled and runtime_available and not smoke_test_on_startup),
        failure_reason=failure_reason,
        device_path=device_path,
        platform=system,
        containerized=containerized,
    )


def _probe_runtime(backend: str, system: str, vaapi_device: str | None) -> tuple[bool, str | None, str | None]:
    if backend.startswith("cpu"):
        return True, None, None
    if backend.startswith("nvidia"):
        probe = probe_nvidia_runtime()
        return probe.nvidia_runtime_available, probe.nvidia_runtime_reason, None
    if backend.startswith("amd_amf"):
        return _probe_amf_runtime(system)
    if backend.startswith("vaapi"):
        device = _resolve_vaapi_device(vaapi_device)
        if not device:
            return False, "/dev/dri/renderD128 không khả dụng", None
        return True, None, device
    if backend.startswith("videotoolbox"):
        if system == "Darwin":
            return True, None, None
        return False, f"VideoToolbox không khả dụng trên {system}", None
    return False, f"Backend không hỗ trợ: {backend}", None


def _probe_amf_runtime(system: str) -> tuple[bool, str | None, str | None]:
    library_names = ["amfrt64.dll", "libamfrt64.so", "libamfrt64.so.1"]
    if system not in {"Windows", "Linux"}:
        return False, f"AMF không khả dụng trên {system}", None
    for library_name in library_names:
        try:
            ctypes.CDLL(library_name)
            return True, None, None
        except OSError:
            continue
    return False, "AMF runtime không khả dụng", None


def _resolve_vaapi_device(vaapi_device: str | None) -> str | None:
    candidate = (vaapi_device or "auto").strip()
    if candidate and candidate != "auto":
        if Path(candidate).exists():
            return candidate
        return None
    for candidate_path in (Path("/dev/dri/renderD128"), Path("/dev/dri/renderD129")):
        if candidate_path.exists():
            return str(candidate_path)
    return None


def _find_vaapi_device() -> str | None:
    return _resolve_vaapi_device("auto")


def _select_auto_backend(
    codec: str,
    *,
    system: str,
    machine: str,
    capabilities: dict[str, EncoderCapability],
    container_gpu_mode: str,
) -> tuple[str, str | None]:
    wants_h265 = codec == "h265"
    cpu_backend = _cpu_backend_for_codec(codec)
    gpu_mode = (container_gpu_mode or "auto").strip().lower()

    if gpu_mode == "cpu":
        return cpu_backend, "container_gpu_mode=cpu"

    candidate_order: list[str] = []
    if system == "Darwin":
        candidate_order = [f"videotoolbox_{'h265' if wants_h265 else 'h264'}"]
    elif system == "Windows":
        candidate_order = [
            f"nvidia_{'h265' if wants_h265 else 'h264'}",
            f"amd_amf_{'h265' if wants_h265 else 'h264'}",
        ]
    else:
        candidate_order = [
            f"nvidia_{'h265' if wants_h265 else 'h264'}",
            f"vaapi_{'h265' if wants_h265 else 'h264'}",
        ]

    if gpu_mode == "nvidia":
        candidate_order = [f"nvidia_{'h265' if wants_h265 else 'h264'}"]
    elif gpu_mode == "vaapi":
        candidate_order = [f"vaapi_{'h265' if wants_h265 else 'h264'}"]

    for backend in candidate_order:
        capability = capabilities.get(backend)
        if capability and capability.smoke_test_passed:
            return backend, None

    return cpu_backend, f"Không tìm thấy backend hardware hợp lệ cho {system}"


def _cpu_backend_for_codec(codec: str) -> str:
    return "cpu_h265" if codec == "h265" else "cpu_h264"


def _encoder_args(backend: str, encoder: str, preset: str, width: int, height: int) -> list[str]:
    preset = preset if preset in {"fast", "balanced", "quality"} else "balanced"
    if backend.startswith("cpu"):
        return _cpu_encoder_args(encoder, preset)
    if backend.startswith("nvidia"):
        return _nvidia_encoder_args(encoder, preset)
    if backend.startswith("amd_amf"):
        return _amf_encoder_args(encoder, preset)
    if backend.startswith("vaapi"):
        return _vaapi_encoder_args(encoder, preset, width, height)
    if backend.startswith("videotoolbox"):
        return _videotoolbox_encoder_args(encoder, preset, width, height)
    return _cpu_encoder_args("libx264", preset)


def _cpu_encoder_args(encoder: str, preset: str) -> list[str]:
    crf_map = {"fast": "23", "balanced": "20", "quality": "18"}
    preset_map = {"fast": "veryfast", "balanced": "medium", "quality": "slow"}
    return ["-c:v", encoder, "-preset", preset_map[preset], "-crf", crf_map[preset]]


def _nvidia_encoder_args(encoder: str, preset: str) -> list[str]:
    cq_map = {"fast": "25", "balanced": "21", "quality": "18"}
    preset_map = {"fast": "p2", "balanced": "p4", "quality": "p6"}
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


def _amf_encoder_args(encoder: str, preset: str) -> list[str]:
    quality_map = {"fast": "speed", "balanced": "balanced", "quality": "quality"}
    return [
        "-c:v",
        encoder,
        "-quality",
        quality_map[preset],
        "-usage",
        "transcoding",
    ]


def _vaapi_encoder_args(encoder: str, preset: str, width: int, height: int) -> list[str]:
    bitrate = _bitrate_for_resolution(width, height, preset)
    return [
        "-c:v",
        encoder,
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        bitrate,
    ]


def _videotoolbox_encoder_args(encoder: str, preset: str, width: int, height: int) -> list[str]:
    bitrate = _bitrate_for_resolution(width, height, preset)
    return [
        "-c:v",
        encoder,
        "-b:v",
        bitrate,
    ]


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


def _run_ffmpeg_probe(args: list[str]) -> tuple[bool, str | None]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return False, f"Smoke test thất bại: {exc.__class__.__name__}: {exc}"
    if completed.returncode == 0:
        return True, None
    stderr = (completed.stderr or "").strip()
    if stderr:
        for pattern in HARDWARE_FAILURE_PATTERNS:
            if pattern in stderr.lower():
                return False, stderr.splitlines()[-1].strip() if stderr.splitlines() else pattern
        return False, stderr.splitlines()[-1].strip() if stderr.splitlines() else "Smoke test thất bại"
    return False, "Smoke test thất bại"


def _is_containerized() -> bool:
    if os.path.exists("/.dockerenv"):
        return True
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            content = cgroup.read_text(encoding="utf-8", errors="ignore").lower()
            return "docker" in content or "containerd" in content or "kubepods" in content
    except Exception:
        pass
    return False


def _nvidia_device_hint_available(system: str) -> bool:
    if system == "Linux":
        if any(Path(path).exists() for path in ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm")):
            return True
        if Path("/proc/driver/nvidia/version").exists():
            return True
        return False
    if system == "Windows":
        return bool(os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip()) and os.environ.get("NVIDIA_VISIBLE_DEVICES", "").strip() not in {"void", "none"}
    return False
