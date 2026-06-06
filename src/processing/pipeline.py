from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from ffmpeg_tools.encoders import (
    detect_available_encoders,
    mark_backend_unavailable,
    select_encoder,
    extract_hardware_failure_reason,
    is_hardware_runtime_failure,
)
from ffmpeg_tools.filters import (
    build_audio_filter,
    build_base_video_filter,
    build_color_adjust_filter,
    build_lut_filter,
    build_noise_overlay_filter,
    build_segment_video_filter,
    build_speed_filter,
    choose_output_resolution,
)
from ffmpeg_tools.probe import probe_video
from ffmpeg_tools.runner import FFmpegError, run_command
from models import AppConfig, EncoderConfig, EncoderSelection, ProcessResult, VideoInfo, VideoJob, JobSource
from processing.audio import choose_optional_audio_assets, random_eq_values
from processing.metadata import metadata_args
from processing.subtitles import burn_subtitles_into_video, generate_subtitles_for_video
from processing.video import Segment, generate_random_segments, generate_scene_segments
from utils.files import safe_output_path
from utils.timing import elapsed_timer
from utils.runtime_logging import (
    JobRuntimeContext,
    build_job_runtime_context,
    build_synthetic_job_id,
    classification_for_pipeline,
    job_context_scope,
    job_prefix,
    log_runtime_execution_plan,
    print_runtime_execution_plan,
    redact_command,
    resolve_whisper_runtime,
    stage_scope,
    stage_skip,
)

LOGGER = logging.getLogger(__name__)


def process_video(
    input_file: Path,
    *,
    input_root: Path,
    output_root: Path,
    temp_root: Path,
    project_root: Path,
    config: AppConfig,
    lut_paths: list[Path] | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
    job: VideoJob | None = None,
    worker_slot: int | None = None,
    worker_total: int | None = None,
) -> ProcessResult:
    with elapsed_timer() as elapsed:
        output_file: Path | None = None
        work_dir = temp_root / f"{_safe_stem(input_file)}_{abs(hash(str(input_file)))}"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            output_file = safe_output_path(input_file, input_root, output_root)
            job_context = _build_job_context(
                input_file=input_file,
                output_file=output_file,
                job=job,
                worker_slot=worker_slot,
                worker_total=worker_total,
            )
            per_job_logs = getattr(config, "logging", None) and bool(config.logging.per_job_logs)
            per_job_log_dir = project_root / "logs" / "jobs"
            ffmpeg_debug = config.processing.debug_ffmpeg or config.logging.debug_ffmpeg_commands
            whisper_runtime = resolve_whisper_runtime(config.subtitles)

            with job_context_scope(job_context, log_dir=per_job_log_dir, enabled=per_job_logs):
                _log_runtime_start(job_context, config)
                with stage_scope(job_context, "LOAD_CONFIG", logger=LOGGER, config_path="config.yaml"):
                    _ = config
                with stage_scope(job_context, "RESOLVE_PATHS", logger=LOGGER, input_root=input_root, output_root=output_root, temp_root=temp_root):
                    pass
                with stage_scope(job_context, "PROBE_INPUT", logger=LOGGER, start_level=logging.INFO):
                    info = probe_video(input_file)
                _log_probe_info(job_context, info)
                with stage_scope(job_context, "VALIDATE_STREAMS", logger=LOGGER, has_audio=info.has_audio):
                    _validate_probe(job_context, info)
                with stage_scope(job_context, "RESOLVE_OUTPUT_SIZE", logger=LOGGER):
                    encoder, width, height, available_encoders = _prepare_encoder(info, config)
                _log_encoder_details(job_context, config, encoder, available_encoders, whisper_runtime)
                with stage_scope(job_context, "SELECT_ENCODER", logger=LOGGER, encoder=encoder.codec_name):
                    pass
                with stage_scope(job_context, "CREATE_TEMP_DIR", logger=LOGGER, work_dir=work_dir):
                    pass

                segment_mode = config.video.segment_mode.lower().strip()
                if segment_mode in {"random", "scene"}:
                    with stage_scope(job_context, "BUILD_SEGMENT_PLAN", logger=LOGGER, mode=segment_mode):
                        segments = _generate_segments(info.duration, config)
                    try:
                        _run_segmented_pipeline(
                            input_file=input_file,
                            output_file=output_file,
                            info=info,
                            project_root=project_root,
                            config=config,
                            lut_paths=lut_paths or [],
                            work_dir=work_dir,
                            encoder=encoder,
                            width=width,
                            height=height,
                            segments=segments,
                            job_context=job_context,
                            ffmpeg_debug=ffmpeg_debug,
                        )
                    except Exception as segment_exc:
                        LOGGER.warning(
                            "Luồng xử lý theo đoạn thất bại, chuyển sang xử lý một lượt: %s",
                            segment_exc,
                        )
                        _run_single_pass(
                            source_file=input_file,
                            output_file=output_file,
                            info=info,
                            project_root=project_root,
                            config=config,
                            encoder=encoder,
                            width=width,
                            height=height,
                            apply_visual_effects=True,
                            lut_paths=lut_paths or [],
                            job_context=job_context,
                            ffmpeg_debug=ffmpeg_debug,
                        )
                else:
                    _run_single_pass(
                        source_file=input_file,
                        output_file=output_file,
                        info=info,
                        project_root=project_root,
                        config=config,
                        encoder=encoder,
                        width=width,
                        height=height,
                        apply_visual_effects=True,
                        lut_paths=lut_paths or [],
                        job_context=job_context,
                        ffmpeg_debug=ffmpeg_debug,
                    )

                with stage_scope(job_context, "VALIDATE_PROCESSED_OUTPUT", logger=LOGGER):
                    _log_validation_probe(job_context, input_probe=info, output_probe=probe_video(output_file))

                output_file = _handle_subtitles(
                    output_file=output_file,
                    project_root=project_root,
                    temp_root=temp_root,
                    config=config,
                    encoder=encoder,
                    progress_callback=progress_callback,
                    job_context=job_context,
                    ffmpeg_debug=ffmpeg_debug,
                )

                if config.processing.delete_source:
                    with stage_scope(job_context, "WRITE_METADATA", logger=LOGGER, delete_source=True):
                        input_file.unlink()

                with stage_scope(job_context, "CLEANUP_TEMP", logger=LOGGER):
                    pass
                LOGGER.info("Đã xử lý %s -> %s trong %.2fs", input_file, output_file, elapsed())
                LOGGER.info("%s JOB_COMPLETE elapsed=%.2fs", job_prefix(job_context), elapsed())
                return ProcessResult(input_file, output_file, True, elapsed())
        except Exception as exc:
            LOGGER.exception("Lỗi khi xử lý %s", input_file)
            if output_file and output_file.exists():
                output_file.unlink(missing_ok=True)
            return ProcessResult(input_file, output_file, False, elapsed(), str(exc))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


def _prepare_encoder(info: VideoInfo, config: AppConfig) -> tuple[EncoderSelection, int, int, list[str]]:
    width, height = choose_output_resolution(
        info.width,
        info.height,
        config.video.aspect_ratio,
        config.video.target_resolution,
        config.video.keep_original_resolution or config.video.mode == "original",
    )
    available_encoders = detect_available_encoders()
    encoder = select_encoder(
        config.encoder,
        available_encoders,
        width=width,
        height=height,
        allow_cpu_fallback=config.encoder.allow_cpu_fallback,
        smoke_test_on_startup=config.encoder.smoke_test_on_startup,
        cache_capability_results=config.encoder.cache_capability_results,
        container_gpu_mode=config.runtime.container_gpu_mode if config.runtime.prefer_native_hardware_acceleration else "cpu",
        vaapi_device=config.vaapi.device,
    )
    return encoder, width, height, available_encoders


def _run_segmented_pipeline(
    *,
    input_file: Path,
    output_file: Path,
    info: VideoInfo,
    project_root: Path,
    config: AppConfig,
    work_dir: Path,
    encoder: EncoderSelection,
    width: int,
    height: int,
    lut_paths: list[Path],
    segments: list[Segment],
    job_context: JobRuntimeContext,
    ffmpeg_debug: bool,
) -> None:
    if len(segments) <= 1:
        raise ValueError("Không đủ đoạn để chạy chế độ segment")

    segments_dir = work_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    segment_files: list[Path] = []

    segment_mode = (config.logging.segment_log_mode or "summary").strip().lower()
    segment_started = time.monotonic()
    if segment_mode != "none":
        LOGGER.info("%s SEGMENTS Started | count=%s", job_prefix(job_context), len(segments))

    for segment in segments:
        segment_file = segments_dir / f"segment_{segment.index:03d}.mp4"
        segment_files.append(segment_file)
        segment_zoom = _segment_zoom(config, segment.index)
        flip = config.video.alternating_flip and segment.index % 2 == 1
        segment_filter = build_segment_video_filter(
            mode=config.video.mode,
            width=width,
            height=height,
            zoom=segment_zoom,
            horizontal_flip=flip,
            fade_seconds=config.video.fade_seconds,
            duration=segment.end - segment.start,
            contrast=config.color.contrast,
            saturation=config.color.saturation,
            sharpen=config.color.sharpen,
            noise_overlay=config.video.noise_overlay,
            noise_alpha=config.video.noise_alpha,
        )
        with stage_scope(
            job_context,
            f"SEGMENT {segment.index}/{len(segments)}",
            logger=LOGGER,
            start=f"{segment.start:.3f}s",
            end=f"{segment.end:.3f}s",
            zoom=f"{segment_zoom:.3f}",
            flip=str(flip).lower(),
        ):
            _render_segment(
                source_file=input_file,
                segment=segment,
                segment_file=segment_file,
                segment_filter=segment_filter,
                info=info,
                config=config,
                encoder=encoder,
                job_context=job_context,
                debug=ffmpeg_debug,
                stderr_tail_lines=config.logging.ffmpeg_stderr_tail_lines,
            )
        if segment_mode == "summary" and segment.index % 5 == 0:
            LOGGER.info("%s SEGMENTS Progress %s/%s", job_prefix(job_context), segment.index, len(segments))

    concat_source = work_dir / "concat.txt"
    with stage_scope(job_context, "CONCAT_SEGMENTS", logger=LOGGER, concat_list=str(concat_source)):
        _write_concat_file(concat_source, segment_files)
        concat_output = work_dir / "concat.mp4"
        _concat_segments(concat_source, concat_output, ffmpeg_debug, config.logging.ffmpeg_stderr_tail_lines)

    concat_info = probe_video(concat_output)
    _run_single_pass(
        source_file=concat_output,
        output_file=output_file,
        info=concat_info,
        project_root=project_root,
        config=config,
        encoder=encoder,
        width=width,
        height=height,
        apply_visual_effects=False,
        lut_paths=lut_paths,
        job_context=job_context,
        ffmpeg_debug=ffmpeg_debug,
    )
    if segment_mode != "none":
        LOGGER.info(
            "%s SEGMENTS Done | %s/%s | %.2fs",
            job_prefix(job_context),
            len(segments),
            len(segments),
            time.monotonic() - segment_started,
        )


def _render_segment(
    *,
    source_file: Path,
    segment: Segment,
    segment_file: Path,
    segment_filter: str,
    info: VideoInfo,
    config: AppConfig,
    encoder: EncoderSelection,
    job_context: JobRuntimeContext,
    debug: bool,
    stderr_tail_lines: int,
) -> None:
    video_filter = f"[0:v]{segment_filter}[vout]"
    if encoder.backend.startswith("vaapi"):
        video_filter = f"[0:v]{segment_filter},format=nv12,hwupload[vout]"
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-ss",
        f"{segment.start:.6f}",
        "-i",
        str(source_file),
        "-t",
        f"{segment.end - segment.start:.6f}",
        "-filter_complex",
        video_filter,
        "-map",
        "[vout]",
    ]
    if info.has_audio:
        args.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"])
    else:
        args.append("-an")

    args.extend(encoder.args)
    args.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(segment_file)])
    try:
        run_command(args, debug=debug, stderr_tail_lines=stderr_tail_lines)
    except FFmpegError as exc:
        if encoder.backend.startswith("cpu") or not is_hardware_runtime_failure(exc.result.stderr):
            raise
        reason = extract_hardware_failure_reason(exc.result.stderr)
        LOGGER.warning(
            "%s [SEGMENT %s] Hardware backend %s failed: %s | retrying with CPU",
            job_prefix(job_context),
            segment.index,
            encoder.backend,
            reason,
        )
        mark_backend_unavailable(encoder.backend, reason)
        if segment_file.exists():
            segment_file.unlink(missing_ok=True)
        cpu_backend = "cpu_h265" if config.encoder.codec == "h265" else "cpu_h264"
        cpu_encoder = select_encoder(
            EncoderConfig(
                backend=cpu_backend,
                codec=config.encoder.codec,
                preset=config.encoder.preset,
                pix_fmt=config.encoder.pix_fmt,
                faststart=config.encoder.faststart,
                allow_cpu_fallback=True,
                smoke_test_on_startup=False,
                cache_capability_results=False,
            ),
            detect_available_encoders(),
            allow_cpu_fallback=True,
            smoke_test_on_startup=False,
            cache_capability_results=False,
            container_gpu_mode="cpu",
        )
        retry_video_filter = f"[0:v]{segment_filter}[vout]"
        args = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-ss",
            f"{segment.start:.6f}",
            "-i",
            str(source_file),
            "-t",
            f"{segment.end - segment.start:.6f}",
            "-filter_complex",
            retry_video_filter,
            "-map",
            "[vout]",
        ]
        if info.has_audio:
            args.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "192k"])
        else:
            args.append("-an")
        args.extend(cpu_encoder.args)
        args.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(segment_file)])
        run_command(args, debug=debug, stderr_tail_lines=stderr_tail_lines)


def _write_concat_file(concat_source: Path, segment_files: list[Path]) -> None:
    lines = [f"file '{path.as_posix()}'" for path in segment_files]
    concat_source.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _concat_segments(concat_source: Path, concat_output: Path, debug: bool, stderr_tail_lines: int) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_source),
        "-c",
        "copy",
        str(concat_output),
    ]
    run_command(args, debug=debug, stderr_tail_lines=stderr_tail_lines)


def _run_single_pass(
    *,
    source_file: Path,
    output_file: Path,
    info: VideoInfo,
    project_root: Path,
    config: AppConfig,
    encoder: EncoderSelection,
    width: int,
    height: int,
    apply_visual_effects: bool,
    lut_paths: list[Path],
    job_context: JobRuntimeContext,
    ffmpeg_debug: bool,
) -> None:
    ambient, bgm = choose_optional_audio_assets(project_root, config.audio)
    with stage_scope(job_context, "BUILD_VIDEO_FILTERS", logger=LOGGER, mode=config.video.mode):
        filter_complex, maps = _build_filter_complex(
            info=info,
            width=width,
            height=height,
            encoder_backend=encoder.backend,
            lut_paths=lut_paths,
            ambient=ambient,
            bgm=bgm,
            config=config,
            apply_visual_effects=apply_visual_effects,
            job_context=job_context,
        )

    args = ["ffmpeg", "-y", "-hide_banner", "-i", str(source_file)]
    if ambient:
        args.extend(["-stream_loop", "-1", "-i", str(ambient)])
    if bgm:
        args.extend(["-stream_loop", "-1", "-i", str(bgm)])

    args.extend(["-filter_complex", filter_complex])
    for stream_map in maps:
        args.extend(["-map", stream_map])

    args.extend(encoder.args)
    args.extend(["-pix_fmt", "yuv420p"])
    if "[aout]" in maps:
        args.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        args.append("-an")
    if config.encoder.faststart:
        args.extend(["-movflags", "+faststart"])
    with stage_scope(job_context, "WRITE_METADATA", logger=LOGGER, metadata_mode=config.metadata.mode):
        args.extend(metadata_args(config.metadata))
    args.append(str(output_file))

    try:
        with stage_scope(job_context, "FINAL_VIDEO_ENCODE", logger=LOGGER, encoder=encoder.codec_name, start_level=logging.INFO):
            LOGGER.info(
                "%s [FINAL_VIDEO_ENCODE] output_resolution=%sx%s aspect=%s mode=%s lut=%s audio=%s",
                job_prefix(job_context),
                width,
                height,
                config.video.aspect_ratio,
                config.video.mode,
                ", ".join(path.name for path in lut_paths) or "none",
                "aout" if "[aout]" in maps else "copy/none",
            )
            LOGGER.debug("%s [FINAL_VIDEO_ENCODE] video_filter_chain=%s", job_prefix(job_context), filter_complex)
            LOGGER.debug("%s [FINAL_VIDEO_ENCODE] ffmpeg_args=%s", job_prefix(job_context), redact_command(args))
            run_command(args, debug=ffmpeg_debug, stderr_tail_lines=config.logging.ffmpeg_stderr_tail_lines)
    except FFmpegError as exc:
        if output_file.exists():
            output_file.unlink(missing_ok=True)
        if encoder.backend.startswith("cpu") or not is_hardware_runtime_failure(exc.result.stderr):
            raise
        reason = extract_hardware_failure_reason(exc.result.stderr)
        LOGGER.warning(
            "%s [FINAL_VIDEO_ENCODE] Hardware backend %s failed: %s | retrying with CPU",
            job_prefix(job_context),
            encoder.backend,
            reason,
        )
        mark_backend_unavailable(encoder.backend, reason)
        cpu_backend = "cpu_h265" if config.encoder.codec == "h265" else "cpu_h264"
        cpu_encoder = select_encoder(
            EncoderConfig(
                backend=cpu_backend,
                codec=config.encoder.codec,
                preset=config.encoder.preset,
                pix_fmt=config.encoder.pix_fmt,
                faststart=config.encoder.faststart,
                allow_cpu_fallback=True,
                smoke_test_on_startup=False,
                cache_capability_results=False,
            ),
            detect_available_encoders(),
            allow_cpu_fallback=True,
            smoke_test_on_startup=False,
            cache_capability_results=False,
            container_gpu_mode="cpu",
        )
        args = ["ffmpeg", "-y", "-hide_banner", "-i", str(source_file)]
        if ambient:
            args.extend(["-stream_loop", "-1", "-i", str(ambient)])
        if bgm:
            args.extend(["-stream_loop", "-1", "-i", str(bgm)])

        filter_complex, maps = _build_filter_complex(
            info=info,
            width=width,
            height=height,
            encoder_backend=cpu_encoder.backend,
            lut_paths=lut_paths,
            ambient=ambient,
            bgm=bgm,
            config=config,
            apply_visual_effects=apply_visual_effects,
            job_context=job_context,
        )
        args.extend(["-filter_complex", filter_complex])
        for stream_map in maps:
            args.extend(["-map", stream_map])
        args.extend(cpu_encoder.args)
        args.extend(["-pix_fmt", "yuv420p"])
        if "[aout]" in maps:
            args.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            args.append("-an")
        if config.encoder.faststart:
            args.extend(["-movflags", "+faststart"])
        with stage_scope(job_context, "WRITE_METADATA", logger=LOGGER, metadata_mode=config.metadata.mode):
            args.extend(metadata_args(config.metadata))
        args.append(str(output_file))
        run_command(args, debug=ffmpeg_debug, stderr_tail_lines=config.logging.ffmpeg_stderr_tail_lines)


def _build_filter_complex(
    *,
    info: VideoInfo,
    width: int,
    height: int,
    encoder_backend: str,
    lut_paths: list[Path],
    ambient: Path | None,
    bgm: Path | None,
    config: AppConfig,
    apply_visual_effects: bool,
    job_context: JobRuntimeContext,
) -> tuple[str, list[str]]:
    parts: list[str] = []
    maps = ["[vout]"]
    output_duration = info.duration / config.video.speed if config.video.speed > 0 else info.duration

    video_filters: list[str] = []
    if apply_visual_effects:
        with stage_scope(job_context, "APPLY_CROP_OR_BLUR", logger=LOGGER, mode=config.video.mode):
            video_filters.append(build_base_video_filter(config.video.mode, width, height))
    else:
        stage_skip(job_context, "APPLY_CROP_OR_BLUR", "apply_visual_effects=false", logger=LOGGER)

    with stage_scope(job_context, "APPLY_COLOR_AND_LUT", logger=LOGGER, lut_count=len(lut_paths)):
        if apply_visual_effects:
            video_filters.append(
                build_color_adjust_filter(config.color.contrast, config.color.saturation, config.color.sharpen)
            )
            if config.video.noise_overlay:
                video_filters.append(build_noise_overlay_filter(config.video.noise_alpha))
        if lut_paths:
            video_filters.append(build_lut_filter(lut_paths))
        elif not config.color.lut_enabled:
            stage_skip(job_context, "APPLY_COLOR_AND_LUT", "color.lut_enabled=false", logger=LOGGER)

    if config.video.speed != 1.0:
        with stage_scope(job_context, "APPLY_SPEED", logger=LOGGER, speed=config.video.speed):
            video_filters.append(build_speed_filter(config.video.speed))
    else:
        stage_skip(job_context, "APPLY_SPEED", "speed=1.0", logger=LOGGER)

    video_filters.append("format=yuv420p")
    parts.append(f"[0:v]{','.join(video_filters)}[vout]")
    if encoder_backend.startswith("vaapi"):
        parts.append("[vout]format=nv12,hwupload[vout_hw]")
        maps = ["[vout_hw]"]
    LOGGER.info("%s [BUILD_VIDEO_FILTERS] video_filter_chain=%s", job_prefix(job_context), ",".join(video_filters))

    audio_labels: list[str] = []
    if info.has_audio:
        with stage_scope(job_context, "PROCESS_AUDIO", logger=LOGGER, has_audio=True):
            bass, treble = random_eq_values(config.audio) if config.audio.random_eq else (0, 0)
            audio_filter = build_audio_filter(
                volume=config.audio.volume,
                speed=config.video.speed,
                tempo_match_speed=config.audio.tempo_match_speed,
                pitch_shift_semitones=config.audio.pitch_shift_semitones,
                random_eq=config.audio.random_eq,
                bass_gain=bass,
                treble_gain=treble,
            )
            parts.append(f"[0:a]{audio_filter}[a0]")
            audio_labels.append("[a0]")
    else:
        stage_skip(job_context, "PROCESS_AUDIO", "no audio stream", logger=LOGGER)

    next_input_index = 1
    if ambient:
        with stage_scope(job_context, "MIX_AMBIENT_OR_BGM", logger=LOGGER, source="ambient"):
            parts.append(
                f"[{next_input_index}:a]atrim=0:{output_duration},asetpts=N/SR/TB,"
                f"volume={config.audio.ambient_volume}[amb]"
            )
            audio_labels.append("[amb]")
            next_input_index += 1
    if bgm:
        with stage_scope(job_context, "MIX_AMBIENT_OR_BGM", logger=LOGGER, source="bgm"):
            parts.append(
                f"[{next_input_index}:a]atrim=0:{output_duration},asetpts=N/SR/TB,"
                f"volume={config.audio.bgm_volume}[bgm]"
            )
            audio_labels.append("[bgm]")

    if audio_labels:
        if len(audio_labels) == 1:
            parts.append(f"{audio_labels[0]}anull[aout]")
        else:
            joined = "".join(audio_labels)
            parts.append(
                f"{joined}amix=inputs={len(audio_labels)}:duration=longest:"
                f"dropout_transition=2,atrim=0:{output_duration},asetpts=N/SR/TB[aout]"
            )
        maps.append("[aout]")

    filter_complex = ";".join(parts)
    LOGGER.debug("%s [BUILD_VIDEO_FILTERS] full_filter_complex=%s", job_prefix(job_context), filter_complex)
    return filter_complex, maps


def _generate_segments(duration: float, config: AppConfig) -> list[Segment]:
    mode = config.video.segment_mode.lower().strip()
    if mode == "scene":
        return generate_scene_segments(
            duration,
            min_seconds=config.video.min_segment_seconds,
            max_seconds=config.video.max_segment_seconds,
        )
    return generate_random_segments(
        duration,
        min_seconds=config.video.min_segment_seconds,
        max_seconds=config.video.max_segment_seconds,
    )


def _segment_zoom(config: AppConfig, index: int) -> float:
    zoom_choices = config.video.alternating_zoom or [1.0]
    multiplier = zoom_choices[index % len(zoom_choices)]
    return max(1.0, config.video.base_zoom * multiplier)


def _safe_stem(path: Path) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem)
    return safe or "video"


def _build_job_context(
    *,
    input_file: Path,
    output_file: Path | None,
    job: VideoJob | None,
    worker_slot: int | None,
    worker_total: int | None,
) -> JobRuntimeContext:
    if job is not None:
        return build_job_runtime_context(
            job_id=job.job_id,
            source=job.source.value,
            input_path=input_file,
            output_path=output_file,
            worker_slot=worker_slot,
            worker_total=worker_total,
        )
    return build_job_runtime_context(
        job_id=build_synthetic_job_id(input_file),
        source=JobSource.LOCAL_INPUT.value,
        input_path=input_file,
        output_path=output_file,
        worker_slot=worker_slot,
        worker_total=worker_total,
    )


def _log_runtime_start(job_context: JobRuntimeContext, config: AppConfig) -> None:
    LOGGER.info(
        "%s Runtime start | source=%s | input=%s | output=%s | worker=%s | thread=%s | pid=%s",
        job_prefix(job_context),
        job_context.source,
        job_context.input_name,
        job_context.output_name,
        job_context.worker_info,
        job_context.thread_name,
        job_context.pid,
    )
    LOGGER.debug(
        "%s Runtime config | ffmpeg_debug=%s | per_job_logs=%s | retain_failed_temp=%s | progress_interval=%ss",
        job_prefix(job_context),
        config.processing.debug_ffmpeg or config.logging.debug_ffmpeg_commands,
        config.logging.per_job_logs,
        config.logging.retain_failed_temp,
        config.logging.progress_interval_seconds,
    )


def _log_probe_info(job_context: JobRuntimeContext, info: VideoInfo) -> None:
    LOGGER.info(
        "%s [PROBE_INPUT] video=%s duration=%.3fs size=%sx%s fps=%.3f codec=%s audio=%s",
        job_prefix(job_context),
        info.path.name,
        info.duration,
        info.width,
        info.height,
        info.fps,
        info.video_codec or "unknown",
        "yes" if info.has_audio else "no",
    )
    LOGGER.debug(
        "%s [PROBE_INPUT] sar=%s dar=%s time_base=%s audio_codec=%s audio_sr=%s audio_ch=%s audio_bitrate=%s audio_duration=%.3f video_bitrate=%s",
        job_prefix(job_context),
        info.sample_aspect_ratio or "unknown",
        info.display_aspect_ratio or "unknown",
        info.time_base or "unknown",
        info.audio_codec or "none",
        info.audio_sample_rate or 0,
        info.audio_channels or 0,
        info.audio_bitrate or 0,
        info.audio_duration or 0.0,
        info.video_bitrate or 0,
    )


def _validate_probe(job_context: JobRuntimeContext, info: VideoInfo) -> None:
    if info.duration <= 0:
        raise RuntimeError("Video duration không hợp lệ")
    if info.width <= 0 or info.height <= 0:
        raise RuntimeError("Kích thước video không hợp lệ")


def _log_encoder_details(
    job_context: JobRuntimeContext,
    config: AppConfig,
    encoder: EncoderSelection,
    available_encoders: list[str],
    whisper_runtime,
) -> None:
    pipeline_classification = classification_for_pipeline(encoder, whisper_runtime)
    fallback_reason = encoder.fallback_reason
    if encoder.requested_backend == "auto" and encoder.backend.startswith("cpu"):
        fallback_reason = fallback_reason or "h264_nvenc và h264_videotoolbox không khả dụng"
    log_runtime_execution_plan(
        job_context,
        config,
        encoder,
        whisper_runtime,
        available_encoders,
        hardware_decoding="NO",
        pipeline_classification=pipeline_classification,
        subtitle_burn_backend="CPU libass + video re-encode",
        video_filters_backend="CPU - scale/crop/blur/LUT/overlay",
        audio_backend="CPU - atempo/volume/EQ",
        fallback_reason=fallback_reason,
    )
    if config.logging.show_runtime_plan:
        print_runtime_execution_plan(
            job_context,
            config,
            encoder,
            whisper_runtime,
            available_encoders,
            hardware_decoding="NO",
            pipeline_classification=pipeline_classification,
            subtitle_burn_backend="CPU libass + video re-encode",
            video_filters_backend="CPU - scale/crop/blur/LUT/overlay",
            audio_backend="CPU - atempo/volume/EQ",
            fallback_reason=fallback_reason,
        )


def _log_validation_probe(job_context: JobRuntimeContext, *, input_probe: VideoInfo, output_probe: VideoInfo) -> None:
    LOGGER.info(
        "%s [VALIDATE_PROCESSED_OUTPUT] before=%sx%s after=%sx%s dar_before=%s dar_after=%s audio_before=%s audio_after=%s duration_before=%.3f duration_after=%.3f",
        job_prefix(job_context),
        input_probe.width,
        input_probe.height,
        output_probe.width,
        output_probe.height,
        input_probe.display_aspect_ratio or f"{input_probe.width}:{input_probe.height}",
        output_probe.display_aspect_ratio or f"{output_probe.width}:{output_probe.height}",
        input_probe.audio_codec or "none",
        output_probe.audio_codec or "none",
        input_probe.duration,
        output_probe.duration,
    )
    if (input_probe.width, input_probe.height) != (output_probe.width, output_probe.height):
        LOGGER.warning("%s Kích thước output thay đổi sau render | before=%sx%s after=%sx%s", job_prefix(job_context), input_probe.width, input_probe.height, output_probe.width, output_probe.height)
    if input_probe.sample_aspect_ratio and output_probe.sample_aspect_ratio and input_probe.sample_aspect_ratio != output_probe.sample_aspect_ratio:
        LOGGER.warning(
            "%s Sample aspect ratio thay đổi | before=%s after=%s",
            job_prefix(job_context),
            input_probe.sample_aspect_ratio,
            output_probe.sample_aspect_ratio,
        )
    if abs(input_probe.duration - output_probe.duration) > 0.5:
        LOGGER.warning(
            "%s Duration thay đổi đáng kể | before=%.3f after=%.3f",
            job_prefix(job_context),
            input_probe.duration,
            output_probe.duration,
        )


def _handle_subtitles(
    *,
    output_file: Path,
    project_root: Path,
    temp_root: Path,
    config: AppConfig,
    encoder: EncoderSelection,
    progress_callback: Callable[[str, str], None] | None = None,
    job_context: JobRuntimeContext | None = None,
    ffmpeg_debug: bool = False,
) -> Path:
    try:
        _emit_progress(progress_callback, "generating_subtitles", "📝 Đang tạo phụ đề...")
        subtitle_result = generate_subtitles_for_video(
            output_file,
            config,
            project_root,
            temp_root,
            output_stem=output_file.stem,
            debug=ffmpeg_debug,
            job_context=job_context,
        )
        if not subtitle_result:
            return output_file

        if subtitle_result.srt_path:
            LOGGER.info("Phụ đề đầu ra: %s", subtitle_result.srt_path)
        if subtitle_result.vtt_path:
            LOGGER.info("Phụ đề VTT đầu ra: %s", subtitle_result.vtt_path)
        if subtitle_result.ass_path:
            LOGGER.info("Phụ đề ASS đầu ra: %s", subtitle_result.ass_path)

        if not config.subtitles.burn_in:
            LOGGER.info("Burn captions đã tắt, giữ nguyên video: %s", output_file)
            return output_file

        subtitle_for_burn = subtitle_result.ass_path or subtitle_result.srt_path
        if not subtitle_for_burn:
            LOGGER.warning("Không có file phụ đề hợp lệ để burn captions cho %s", output_file)
            return output_file

        _emit_progress(progress_callback, "burning_subtitles", "🔥 Đang burn phụ đề vào video...")
        burned_output = output_file.with_name(f"{output_file.stem}_burned{output_file.suffix}")
        burned_output = burned_output if not burned_output.exists() else _unique_burn_output(burned_output)
        burn_subtitles_into_video(
            output_file,
            subtitle_for_burn,
            burned_output,
            config,
            project_root=project_root,
            temp_root=temp_root,
            encoder_selection=encoder,
            debug=ffmpeg_debug,
            job_context=job_context,
            subtitle_result=subtitle_result,
        )
        subtitle_result.burned_output_path = burned_output
        LOGGER.info("Video phụ đề đã burn: %s", burned_output)
        return burned_output
    except Exception as exc:
        LOGGER.exception("Lỗi khi xử lý phụ đề cho %s", output_file)
        LOGGER.error("Chi tiết lỗi phụ đề: %s", exc)
        return output_file


def _emit_progress(
    progress_callback: Callable[[str, str], None] | None,
    stage: str,
    text: str,
) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage, text)
    except Exception:
        LOGGER.exception("Lỗi khi phát sự kiện progress phụ đề | stage=%s", stage)


def _unique_burn_output(path: Path) -> Path:
    counter = 1
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate
