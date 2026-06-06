from __future__ import annotations

import logging
from pathlib import Path

from config import default_config
from ffmpeg_tools.encoders import select_encoder
from models import EncoderConfig, VideoInfo
from processing.pipeline import _log_validation_probe
from utils.runtime_logging import (
    JobRuntimeContext,
    WhisperRuntimeSelection,
    classification_for_pipeline,
    job_context_scope,
    redact_command,
    resolve_whisper_runtime,
    stage_scope,
)


def test_pipeline_classification_distinguishes_gpu_encode_and_cpu_filters() -> None:
    encoder = select_encoder(EncoderConfig(backend="nvidia_h264"), ["h264_nvenc"])
    whisper = WhisperRuntimeSelection("auto", "cuda", "auto", "float16")
    assert classification_for_pipeline(encoder, whisper) == "GPU encode + GPU transcription + CPU filters"

    whisper_cpu = WhisperRuntimeSelection("auto", "cpu", "auto", "int8")
    assert classification_for_pipeline(encoder, whisper_cpu) == "GPU encode + CPU filters"

    cpu_encoder = select_encoder(EncoderConfig(backend="cpu_h264"), ["libx264"])
    assert classification_for_pipeline(cpu_encoder, whisper_cpu) == "CPU-only"


def test_redact_command_hides_tokens_and_signed_urls() -> None:
    command = [
        "ffmpeg",
        "-i",
        "https://api.telegram.org/bot123456:SECRET/sendDocument?signature=abc&expires=1",
        "--header",
        "x-api-key: REVID_API_KEY",
    ]
    redacted = redact_command(command)
    assert "SECRET" not in redacted
    assert "REVID_API_KEY" not in redacted
    assert "[REDACTED]" in redacted


def test_stage_logger_emits_start_done_and_failed(caplog) -> None:
    caplog.set_level(logging.INFO)
    context = JobRuntimeContext(
        job_id="job_test",
        source="local_input",
        input_path=Path("input/video.mp4"),
        output_path=Path("output/video.mp4"),
        worker_slot=1,
        worker_total=2,
        thread_name="worker-1",
        pid=123,
    )

    with job_context_scope(context):
        with stage_scope(context, "PROBE_INPUT", logger=logging.getLogger("test"), start_level=logging.INFO):
            pass

    assert "START" in caplog.text
    assert "DONE" in caplog.text

    caplog.clear()
    with job_context_scope(context):
        try:
            with stage_scope(context, "FINAL_VIDEO_ENCODE", logger=logging.getLogger("test"), start_level=logging.INFO):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
    assert "FAILED" in caplog.text
    assert "elapsed=" in caplog.text


def test_redaction_filters_strip_duplicate_prefix_and_secrets(caplog) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("redaction-test")
    logger.info("[JOB job_123][LOAD_CONFIG] https://api.telegram.org/bot123456:SECRET/getUpdates?signature=abc&expires=1 x-api-key: abc123")

    assert "bot123456:SECRET" not in caplog.text
    assert "abc123" not in caplog.text
    assert "[JOB job_123]" not in caplog.text
    assert "getUpdates" in caplog.text


def test_per_job_log_file_is_created(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.INFO)
    context = JobRuntimeContext(
        job_id="job_file",
        source="local_input",
        input_path=Path("input/video.mp4"),
        output_path=Path("output/video.mp4"),
        worker_slot=1,
        worker_total=1,
        thread_name="worker-1",
        pid=123,
    )
    job_log_dir = tmp_path / "logs" / "jobs"
    logger = logging.getLogger("job-file-test")
    with job_context_scope(context, log_dir=job_log_dir, enabled=True, level=logging.INFO):
        logger.info("hello from job")
    log_file = job_log_dir / "job_file.log"
    assert log_file.exists()
    assert "hello from job" in log_file.read_text(encoding="utf-8")


def test_whisper_runtime_resolution_falls_back_to_cpu_on_invalid_device() -> None:
    config = default_config().subtitles
    config.device = "not-a-device"
    runtime = resolve_whisper_runtime(config)
    assert runtime.resolved_device == "cpu"
    assert runtime.resolved_compute_type == "int8"


def test_validation_probe_logs_changed_dimensions(caplog) -> None:
    caplog.set_level(logging.INFO)
    before = VideoInfo(Path("before.mp4"), 10.0, 1080, 1920, 30.0, True, display_aspect_ratio="9:16")
    after = VideoInfo(Path("after.mp4"), 10.0, 720, 1280, 30.0, True, display_aspect_ratio="9:16")
    context = JobRuntimeContext(
        job_id="job_probe",
        source="local_input",
        input_path=Path("input/video.mp4"),
        output_path=Path("output/video.mp4"),
        worker_slot=1,
        worker_total=1,
        thread_name="worker-1",
        pid=123,
    )
    _log_validation_probe(context, input_probe=before, output_probe=after)
    assert "Kích thước output thay đổi" in caplog.text
