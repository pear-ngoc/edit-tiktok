from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
import os
import platform
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from models import AppConfig, EncoderSelection, JobSource, SubtitlesConfig

LOGGER = logging.getLogger(__name__)

_JOB_ID = contextvars.ContextVar("job_id", default="-")
_WORKER_SLOT = contextvars.ContextVar("worker_slot", default="-")
_WORKER_TOTAL = contextvars.ContextVar("worker_total", default="-")
_STAGE = contextvars.ContextVar("job_stage", default="-")


class NormalizedMessageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = _normalize_message(message)
        record.args = ()
        return True


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.__dict__.setdefault("job_id", "-")
        record.__dict__.setdefault("worker_info", "-")
        record.__dict__.setdefault("worker_display", "")
        record.__dict__.setdefault("stage", "-")
        rendered = super().format(record)
        return _redact_text(rendered)


@dataclass(slots=True)
class JobRuntimeContext:
    job_id: str
    source: str
    input_path: Path
    output_path: Path | None
    worker_slot: int | None = None
    worker_total: int | None = None
    thread_name: str = ""
    pid: int = 0

    @property
    def worker_info(self) -> str:
        if self.worker_slot is None:
            return "-"
        if self.worker_total is None:
            return str(self.worker_slot)
        return f"{self.worker_slot}/{self.worker_total}"

    @property
    def input_name(self) -> str:
        return self.input_path.name

    @property
    def output_name(self) -> str:
        return self.output_path.name if self.output_path else "-"


@dataclass(slots=True)
class WhisperRuntimeSelection:
    requested_device: str
    resolved_device: str
    requested_compute_type: str
    resolved_compute_type: str
    fallback_reason: str | None = None


class RuntimeContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _JOB_ID.get()
        record.worker_slot = _WORKER_SLOT.get()
        record.worker_total = _WORKER_TOTAL.get()
        record.worker_info = _format_worker_info(record.worker_slot, record.worker_total)
        record.worker_display = f"W{record.worker_info}" if record.worker_info != "-" else ""
        record.stage = _STAGE.get()
        return True


class JobIdFilter(logging.Filter):
    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "job_id", "-") == self.job_id


def install_runtime_context_filter(logger: logging.Logger | None = None) -> None:
    target = logger or logging.getLogger()
    has_filter = any(isinstance(item, RuntimeContextFilter) for item in target.filters)
    if not has_filter:
        target.addFilter(RuntimeContextFilter())


@contextlib.contextmanager
def job_context_scope(
    context: JobRuntimeContext,
    *,
    log_dir: Path | None = None,
    enabled: bool = True,
    level: int = logging.INFO,
) -> Iterator[None]:
    token_job = _JOB_ID.set(context.job_id)
    token_worker_slot = _WORKER_SLOT.set(str(context.worker_slot) if context.worker_slot is not None else "-")
    token_worker_total = _WORKER_TOTAL.set(str(context.worker_total) if context.worker_total is not None else "-")
    token_stage = _STAGE.set("JOB")
    handler: logging.Handler | None = None
    root_logger = logging.getLogger()
    try:
        existing_job_handler = any(
            getattr(item, "_job_log_job_id", None) == context.job_id for item in root_logger.handlers
        )
        if enabled and log_dir is not None and not existing_job_handler:
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_dir / f"{context.job_id}.log", encoding="utf-8")
            handler.setLevel(level)
            handler.setFormatter(_build_formatter())
            handler.addFilter(RuntimeContextFilter())
            handler.addFilter(JobIdFilter(context.job_id))
            setattr(handler, "_job_log_job_id", context.job_id)
            root_logger.addHandler(handler)
        yield
    finally:
        if handler is not None:
            root_logger.removeHandler(handler)
            handler.close()
        _STAGE.reset(token_stage)
        _WORKER_TOTAL.reset(token_worker_total)
        _WORKER_SLOT.reset(token_worker_slot)
        _JOB_ID.reset(token_job)


@contextlib.contextmanager
def stage_scope(context: JobRuntimeContext, stage_name: str, *, logger: logging.Logger | None = None, **details: Any) -> Iterator[None]:
    logger = logger or LOGGER
    token = _STAGE.set(stage_name)
    start_level = int(details.pop("start_level", logging.DEBUG))
    done_level = int(details.pop("done_level", logging.INFO))
    show_start = bool(details.pop("show_start", True))
    if show_start and logger.isEnabledFor(start_level):
        logger.log(start_level, "%s", _format_stage_message(stage_name, "START", **details))
    start_time = os.times()
    try:
        yield
    except Exception as exc:
        elapsed = _elapsed_seconds(start_time)
        failed_details = dict(details)
        failed_details["elapsed"] = f"{elapsed:.2f}s"
        failed_details["error"] = f"{exc.__class__.__name__}: {_safe_text(str(exc))}"
        logger.error("%s", _format_stage_message(stage_name, "FAILED", **failed_details))
        raise
    else:
        elapsed = _elapsed_seconds(start_time)
        done_details = dict(details)
        done_details["elapsed"] = f"{elapsed:.2f}s"
        logger.log(done_level, "%s", _format_stage_message(stage_name, "DONE", **done_details))
    finally:
        _STAGE.reset(token)


def stage_skip(context: JobRuntimeContext, stage_name: str, reason: str, *, logger: logging.Logger | None = None) -> None:
    logger = logger or LOGGER
    logger.info("%s", _format_stage_message(stage_name, "SKIP", reason=reason))


def build_job_runtime_context(
    *,
    job_id: str,
    source: str,
    input_path: Path,
    output_path: Path | None,
    worker_slot: int | None,
    worker_total: int | None,
) -> JobRuntimeContext:
    return JobRuntimeContext(
        job_id=job_id,
        source=source,
        input_path=input_path,
        output_path=output_path,
        worker_slot=worker_slot,
        worker_total=worker_total,
        thread_name=threading.current_thread().name,
        pid=os.getpid(),
    )


def build_synthetic_job_id(input_path: Path) -> str:
    digest = hashlib.sha1(input_path.resolve().as_posix().encode("utf-8")).hexdigest()
    return f"job_{digest[:12]}"


def job_prefix(context: JobRuntimeContext) -> str:
    return f"[JOB {context.job_id}]"


def log_runtime_execution_plan(
    context: JobRuntimeContext,
    config: AppConfig,
    encoder: EncoderSelection,
    whisper_runtime: WhisperRuntimeSelection | None,
    available_encoders: list[str],
    *,
    hardware_decoding: str = "NO",
    pipeline_classification: str = "mixed/unknown",
    subtitle_burn_backend: str = "CPU libass",
    video_filters_backend: str = "CPU",
    audio_backend: str = "CPU",
    fallback_reason: str | None = None,
) -> None:
    logger = LOGGER
    nvenc_runtime = _probe_nvidia_runtime()
    nvenc_runtime_label = _nvidia_runtime_label(nvenc_runtime)
    logger.info("Runtime execution plan")
    logger.info("Platform: %s %s", platform.system(), platform.machine())
    logger.info(
        "Worker: %s | thread=%s | pid=%s",
        context.worker_info,
        context.thread_name or threading.current_thread().name,
        context.pid or os.getpid(),
    )
    logger.info("Input: %s", context.input_name)
    logger.info("Output: %s", context.output_name)
    logger.info("Encoder requested: %s", encoder.requested_backend or config.encoder.backend)
    logger.info("Encoder resolved: %s", encoder.backend)
    logger.info("FFmpeg encoder: %s", encoder.codec_name)
    logger.info("Requested video codec: %s", config.encoder.codec)
    logger.info("Available encoders: %s", ", ".join(sorted(_interesting_encoders(available_encoders))) or "none")
    logger.info("NVIDIA runtime: %s", nvenc_runtime_label)
    logger.info("Hardware encoding: %s", _hardware_encoding_label(encoder))
    logger.info("Hardware decoding: %s", hardware_decoding)
    logger.info("Video filters: %s", video_filters_backend)
    logger.info("Audio filters: %s", audio_backend)
    if whisper_runtime is None:
        logger.info("Faster-Whisper: disabled")
    else:
        logger.info("Faster-Whisper: %s, compute_type=%s", _whisper_device_label(whisper_runtime.resolved_device), whisper_runtime.resolved_compute_type)
        logger.info("Faster-Whisper requested: device=%s compute_type=%s", whisper_runtime.requested_device, whisper_runtime.requested_compute_type)
    logger.info("Subtitle burn backend: %s", subtitle_burn_backend)
    if fallback_reason or encoder.fallback_reason or (whisper_runtime and whisper_runtime.fallback_reason):
        reason = fallback_reason or encoder.fallback_reason or whisper_runtime.fallback_reason
        logger.info("Fallback reason: %s", reason)
    logger.info("Pipeline classification: %s", pipeline_classification)


def print_runtime_execution_plan(
    context: JobRuntimeContext,
    config: AppConfig,
    encoder: EncoderSelection,
    whisper_runtime: WhisperRuntimeSelection | None,
    available_encoders: list[str],
    *,
    hardware_decoding: str = "NO",
    pipeline_classification: str = "mixed/unknown",
    subtitle_burn_backend: str = "CPU libass",
    video_filters_backend: str = "CPU",
    audio_backend: str = "CPU",
    fallback_reason: str | None = None,
) -> None:
    if pipeline_classification == "mixed/unknown":
        pipeline_classification = classification_for_pipeline(
            encoder,
            whisper_runtime,
        )
    nvenc_runtime = _probe_nvidia_runtime()
    nvenc_runtime_label = _nvidia_runtime_label(nvenc_runtime)
    lines = [
        "Runtime execution plan",
        f"Platform: {platform.system()} {platform.machine()}",
        f"Worker: {context.worker_info}, thread={context.thread_name or threading.current_thread().name}, pid={context.pid or os.getpid()}",
        f"Input: {context.input_name}",
        f"Output: {context.output_name}",
        f"Encoder requested: {encoder.requested_backend or config.encoder.backend}",
        f"Encoder resolved: {encoder.backend}",
        f"FFmpeg encoder: {encoder.codec_name}",
        f"Requested video codec: {config.encoder.codec}",
        f"Available encoders: {', '.join(sorted(_interesting_encoders(available_encoders))) or 'none'}",
        f"NVIDIA runtime: {nvenc_runtime_label}",
        f"Hardware encoding: {_hardware_encoding_label(encoder)}",
        f"Hardware decoding: {hardware_decoding}",
        f"Video filters: {video_filters_backend}",
        f"Audio filters: {audio_backend}",
    ]
    if whisper_runtime is None:
        lines.append("Faster-Whisper: disabled")
    else:
        lines.append(f"Faster-Whisper: {_whisper_device_label(whisper_runtime.resolved_device)}, compute_type={whisper_runtime.resolved_compute_type}")
        lines.append(f"Faster-Whisper requested: device={whisper_runtime.requested_device} compute_type={whisper_runtime.requested_compute_type}")
    lines.append(f"Subtitle burn backend: {subtitle_burn_backend}")
    if fallback_reason or encoder.fallback_reason or (whisper_runtime and whisper_runtime.fallback_reason):
        reason = fallback_reason or encoder.fallback_reason or whisper_runtime.fallback_reason
        lines.append(f"Fallback reason: {reason}")
    lines.append(f"Pipeline classification: {pipeline_classification}")
    for line in lines:
        print(line)


def log_startup_summary(
    project_root: Path,
    config: AppConfig,
    *,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
    available_encoders: list[str],
    encoder: EncoderSelection,
    whisper_runtime: WhisperRuntimeSelection | None,
) -> None:
    logger = LOGGER
    nvenc_runtime = _probe_nvidia_runtime()
    nvenc_runtime_label = _nvidia_runtime_label(nvenc_runtime)
    logger.info("[RUNTIME] OS=%s %s | CPU=%s | FFmpeg=%s | FFprobe=%s", platform.system(), platform.machine(), os.cpu_count() or "unknown", ffmpeg_path or "missing", ffprobe_path or "missing")
    logger.info(
        "[RUNTIME] Encoders | default=%s | VideoToolbox build=%s | NVENC build=%s | NVENC runtime=%s | CUDA transcription=%s | workers=%s",
        encoder.codec_name,
        "available" if _has_encoder(available_encoders, "videotoolbox") else "unavailable",
        "available" if _has_encoder(available_encoders, "nvenc") else "unavailable",
        nvenc_runtime_label,
        "available" if whisper_runtime and whisper_runtime.resolved_device == "cuda" else "unavailable",
        config.queue.max_workers,
    )
    if _has_encoder(available_encoders, "nvenc") and not nvenc_runtime.nvidia_runtime_available:
        logger.warning(
            "[RUNTIME] FFmpeg có h264_nvenc/hevc_nvenc nhưng runtime NVIDIA không khả dụng: %s",
            nvenc_runtime.nvidia_runtime_reason or "unknown",
        )
    if config.queue.max_workers >= 5:
        logger.warning("[RUNTIME] Warning: five simultaneous FFmpeg jobs may overload this machine")
    logger.debug("[RUNTIME] Project root: %s", project_root)


def redact_command(args: list[str]) -> str:
    redacted: list[str] = []
    for arg in args:
        redacted.append(_redact_text(arg))
    return " ".join(redacted)


def redact_text(text: str, limit: int = 180) -> str:
    cleaned = _redact_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def resolve_whisper_runtime(config: SubtitlesConfig) -> WhisperRuntimeSelection:
    requested_device = (config.device or "auto").strip().lower()
    requested_compute = (config.compute_type or "auto").strip().lower()
    resolved_device = requested_device
    fallback_reason: str | None = None
    if requested_device == "auto":
        resolved_device = "cpu"
        if platform.system() in {"Windows", "Linux"} and _gpu_likely_available():
            resolved_device = "cuda"
    elif requested_device not in {"cpu", "cuda"}:
        resolved_device = "cpu"
        fallback_reason = f"Thiết bị faster-whisper không hợp lệ: {requested_device}"

    resolved_compute = requested_compute
    if resolved_compute == "auto":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    elif resolved_device == "cuda" and resolved_compute in {"int8", "int8_float16"}:
        fallback_reason = fallback_reason or None
    elif resolved_device == "cpu" and resolved_compute in {"float16", "float32"}:
        resolved_compute = "int8"

    return WhisperRuntimeSelection(
        requested_device=requested_device,
        resolved_device=resolved_device,
        requested_compute_type=requested_compute,
        resolved_compute_type=resolved_compute,
        fallback_reason=fallback_reason,
    )


def classification_for_pipeline(encoder: EncoderSelection, whisper_runtime: WhisperRuntimeSelection | None) -> str:
    hardware_encode = not encoder.backend.startswith("cpu")
    whisper_gpu = whisper_runtime is not None and whisper_runtime.resolved_device == "cuda"
    if not hardware_encode and not whisper_gpu:
        return "CPU-only"
    if hardware_encode and whisper_gpu:
        return "GPU encode + GPU transcription + CPU filters"
    if hardware_encode:
        return "GPU encode + CPU filters"
    if whisper_gpu:
        return "GPU transcription + CPU filters"
    return "mixed/unknown"


def _format_stage_message(stage: str, state: str, **details: Any) -> str:
    extras = _format_details(details)
    label = {
        "START": "Started",
        "DONE": "Done",
        "FAILED": "Failed",
        "SKIP": "Skipped",
    }.get(state, state.title())
    if state == "DONE" and "elapsed" in details:
        label = f"Done in {details['elapsed']}"
        details = {key: value for key, value in details.items() if key != "elapsed"}
        extras = _format_details(details)
    elif state == "FAILED" and "elapsed" in details:
        label = f"Failed after {details['elapsed']}"
        details = {key: value for key, value in details.items() if key != "elapsed"}
        extras = _format_details(details)
    return f"{label}" + (f" | {extras}" if extras else "")


def _format_worker_info(worker_slot: str | int, worker_total: str | int) -> str:
    if str(worker_slot) == "-":
        return "-"
    if str(worker_total) == "-":
        return str(worker_slot)
    return f"{worker_slot}/{worker_total}"


def _format_details(details: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in details.items():
        if value is None:
            continue
        parts.append(f"{key}={_safe_text(str(value))}")
    return " ".join(parts)


def _safe_text(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _build_formatter() -> logging.Formatter:
    return RedactingFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(job_id)s | %(worker_info)s | %(stage)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _build_compact_formatter() -> logging.Formatter:
    return RedactingFormatter(
        "%(asctime)s %(levelname)-5s %(job_id)s %(worker_display)s %(stage)-16s %(message)s",
        datefmt="%H:%M:%S",
    )


def _redact_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"(https?://api\.telegram\.org/bot)([^/\s]+)", r"\1***REDACTED***", text)
    text = re.sub(r"(https?://[^/\s]*tiktok[^/\s]*)", _redact_url_match, text, flags=re.IGNORECASE)
    text = re.sub(r"(x-api-key[:=]\s*)(\S+)", r"\1***REDACTED***", text, flags=re.IGNORECASE)
    text = re.sub(r"(authorization[:=]\s*bearer\s+)(\S+)", r"\1***REDACTED***", text, flags=re.IGNORECASE)
    text = re.sub(r"(TELEGRAM_BOT_TOKEN|REVID_API_KEY)\s*[:=]\s*([^\s]+)", r"\1=***REDACTED***", text)
    if "download-direct" in text or "signature=" in text or "expires=" in text:
        text = _redact_url_text(text)
    return text


def _normalize_message(text: str) -> str:
    if not text:
        return text
    stripped = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", text).strip()
    return _redact_text(stripped)


def _redact_url_match(match: re.Match[str]) -> str:
    return _redact_url_text(match.group(0))


def _redact_url_text(text: str) -> str:
    try:
        parsed = urlsplit(text)
        query = parse_qsl(parsed.query, keep_blank_values=True)
        if query:
            safe_query: list[tuple[str, str]] = []
            for key, value in query:
                lowered = key.lower()
                if lowered in {"signature", "expires", "url", "token", "api_key", "key"}:
                    safe_query.append((key, "***REDACTED***"))
                else:
                    safe_query.append((key, value))
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(safe_query), ""))
    except Exception:
        pass
    if "?" in text:
        return text.split("?", 1)[0] + "?***REDACTED***"
    return text


def _elapsed_seconds(start_times: os.times_result) -> float:
    current = os.times()
    return max(0.0, (current.elapsed - start_times.elapsed))


def _interesting_encoders(encoders: list[str]) -> list[str]:
    wanted = {
        "libx264",
        "libx265",
        "h264_nvenc",
        "hevc_nvenc",
        "h264_videotoolbox",
        "hevc_videotoolbox",
    }
    return [encoder for encoder in encoders if encoder in wanted]


def _nvidia_runtime_label(runtime: EncoderRuntimeProbe) -> str:
    if runtime.nvidia_runtime_available:
        return "available"
    return f"unavailable ({runtime.nvidia_runtime_reason or 'unknown'})"


def _hardware_encoding_label(encoder: EncoderSelection) -> str:
    if encoder.backend.startswith("nvidia"):
        return "YES - NVIDIA NVENC"
    if encoder.backend.startswith("videotoolbox"):
        return "YES - Apple VideoToolbox"
    return "NO"


def _whisper_device_label(device: str) -> str:
    if device.lower() == "cuda":
        return "CUDA"
    return "CPU"


def _has_encoder(encoders: list[str], needle: str) -> bool:
    return any(needle in encoder for encoder in encoders)


def _probe_nvidia_runtime():
    from ffmpeg_tools.encoders import probe_nvidia_runtime

    return probe_nvidia_runtime()


def _gpu_likely_available() -> bool:
    try:
        import shutil

        if shutil.which("nvidia-smi"):
            return True
    except Exception:
        pass
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return False
        import torch  # type: ignore

        return bool(getattr(torch.cuda, "is_available", lambda: False)())
    except Exception:
        return False
