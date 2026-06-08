from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ffmpeg_tools.encoders import (
    detect_available_encoders,
    extract_hardware_failure_reason,
    is_hardware_runtime_failure,
    mark_backend_unavailable,
    select_encoder,
)
from ffmpeg_tools.filters import build_ass_burn_filter, build_subtitles_burn_filter
from ffmpeg_tools.probe import probe_video
from ffmpeg_tools.runner import FFmpegError, run_command
from models import AppConfig, EncoderConfig, EncoderSelection, FormattingConfig, SubtitlesConfig
from processing.transcription import TranscriptionManager
from utils.runtime_logging import (
    JobRuntimeContext,
    job_context_scope,
    job_prefix,
    stage_scope,
    stage_skip,
)

LOGGER = logging.getLogger(__name__)

_PREFERRED_PUNCTUATION = (".", "?", "!", ",", ";", ":")


@dataclass(slots=True)
class SubtitleWord:
    text: str
    start: float
    end: float


@dataclass(slots=True)
class SubtitleCue:
    start: float
    end: float
    text: str
    lines: list[str] = field(default_factory=list)


SubtitleEntry = SubtitleCue


@dataclass(slots=True)
class AssStyleConfig:
    font_name: str = "Arial"
    font_size: int = 58
    outline: int = 5
    shadow: int = 1
    margin_v: int = 140
    alignment: str = "bottom"
    play_res_x: int = 1080
    play_res_y: int = 1920
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    background_color: str = "#000000"
    text_opacity: float = 0.0
    outline_opacity: float = 0.0
    background_opacity: float = 0.35
    box_enabled: bool = True


@dataclass(slots=True)
class SubtitleGenerationResult:
    media_path: Path
    subtitle_dir: Path
    srt_path: Path | None
    vtt_path: Path | None
    ass_path: Path | None
    language: str
    detected_language: str | None = None
    burned_output_path: Path | None = None
    cues: list[SubtitleCue] = field(default_factory=list)


def transcribe_media(
    media_path: Path,
    config: SubtitlesConfig | AppConfig,
    formatting: FormattingConfig | None = None,
) -> list[SubtitleCue]:
    from processing.transcription import TranscriptionManager

    subtitles_config, formatting_config = _resolve_subtitle_configs(config, formatting)
    manager = TranscriptionManager(subtitles_config, None)
    result = manager.transcribe(media_path, subtitles_config.language)
    words: list[SubtitleWord] = [
        SubtitleWord(text=w.text, start=w.start, end=w.end)
        for w in result.words
    ]
    LOGGER.info(
        "Phụ đề thô từ %s | raw_segments=%s | word_timestamps=%s | backend=%s",
        media_path,
        len(result.segments),
        len(words),
        result.backend,
    )
    return split_words_into_caption_cues(
        words,
        max_chars_per_line=formatting_config.max_chars_per_line,
        max_lines=formatting_config.max_lines,
        max_chars_per_cue=formatting_config.max_chars_per_cue,
        max_words_per_cue=formatting_config.max_words_per_cue,
        min_duration=formatting_config.min_duration,
        max_duration=formatting_config.max_duration,
        pause_threshold=formatting_config.pause_threshold,
    )


def split_words_into_caption_cues(
    words: Iterable[SubtitleWord | dict[str, Any] | Any],
    *,
    max_chars_per_line: int = 20,
    max_lines: int = 2,
    max_chars_per_cue: int = 40,
    max_words_per_cue: int = 7,
    min_duration: float = 0.7,
    max_duration: float = 2.6,
    pause_threshold: float = 0.45,
) -> list[SubtitleCue]:
    normalized_words = [
        word
        for word in (_coerce_word(item) for item in words)
        if word is not None and word.text
    ]
    if not normalized_words:
        return []

    cues: list[SubtitleCue] = []
    current: list[SubtitleWord] = []

    for word in normalized_words:
        if not current:
            current.append(word)
            continue

        pause = max(0.0, word.start - current[-1].end)
        tentative = current + [word]
        tentative_text = _join_words(tentative)
        current_duration = current[-1].end - current[0].start

        if _should_break_before_add(
            current=current,
            pause=pause,
            tentative_text=tentative_text,
            current_duration=current_duration,
            max_chars_per_line=max_chars_per_line,
            max_lines=max_lines,
            max_chars_per_cue=max_chars_per_cue,
            max_duration=max_duration,
            pause_threshold=pause_threshold,
        ):
            cues.append(
                _build_caption_cue(
                    current,
                    max_chars_per_line=max_chars_per_line,
                    max_lines=max_lines,
                )
            )
            current = [word]
            continue

        current.append(word)
        current_text = _join_words(current)
        current_duration = current[-1].end - current[0].start
        if (
            len(current) >= max_words_per_cue
            or current_duration >= max_duration
            or _text_length(current_text) >= max_chars_per_cue
            or _would_exceed_caption_lines(current_text, max_chars_per_line, max_lines)
        ):
            cues.append(
                _build_caption_cue(
                    current,
                    max_chars_per_line=max_chars_per_line,
                    max_lines=max_lines,
                )
            )
            current = []
        elif _ends_caption(word.text) and (current_duration >= min_duration or len(current) >= 2):
            cues.append(
                _build_caption_cue(
                    current,
                    max_chars_per_line=max_chars_per_line,
                    max_lines=max_lines,
                )
            )
            current = []

    if current:
        cues.append(
            _build_caption_cue(
                current,
                max_chars_per_line=max_chars_per_line,
                max_lines=max_lines,
            )
        )

    return cues


def wrap_caption_text(
    text: str,
    max_chars_per_line: int = 20,
    max_lines: int = 2,
) -> list[str]:
    lines, _ = _wrap_caption_text(text, max_chars_per_line, max_lines)
    return lines


def write_srt(entries: Iterable[SubtitleCue], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, entry in enumerate(entries, start=1):
        lines.append(str(index))
        lines.append(
            f"{format_srt_timestamp(entry.start)} --> {format_srt_timestamp(entry.end)}"
        )
        lines.extend(_entry_lines(entry))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_vtt(entries: Iterable[SubtitleCue], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["WEBVTT", ""]
    for entry in entries:
        lines.append(
            f"{format_vtt_timestamp(entry.start)} --> {format_vtt_timestamp(entry.end)}"
        )
        lines.extend(_entry_lines(entry))
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_ass(
    cues: Iterable[SubtitleCue],
    output_path: Path,
    style_config: AssStyleConfig,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alignment = _resolve_ass_alignment(style_config.alignment)
    primary_colour = _ass_colour(style_config.text_color, style_config.text_opacity)
    outline_colour = _ass_colour(style_config.outline_color, style_config.outline_opacity)
    background_colour = _ass_colour(style_config.background_color, style_config.background_opacity)
    border_style = 3 if style_config.box_enabled else 1
    style_line = (
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,"
        "Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding"
    )
    style_values = (
        "Style: TikTok,"
        f"{style_config.font_name},"
        f"{style_config.font_size},"
        f"{primary_colour},"
        "&H000000FF,"
        f"{outline_colour},"
        f"{background_colour},"
        "1,0,0,0,100,100,0,0,"
        f"{border_style},"
        f"{style_config.outline},"
        f"{style_config.shadow},"
        f"{alignment},"
        "40,40,"
        f"{style_config.margin_v},"
        "1"
    )
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {style_config.play_res_x}",
        f"PlayResY: {style_config.play_res_y}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        style_line,
        style_values,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        dialogue_text = _escape_ass_text(_entry_text(cue)).replace(chr(10), r"\N")
        lines.append(
            "Dialogue: 0,"
            f"{format_ass_timestamp(cue.start)},"
            f"{format_ass_timestamp(cue.end)},"
            "TikTok,,0,0,0,,"
            f"{dialogue_text}"
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def generate_subtitles_for_video(
    video_path: Path,
    config: SubtitlesConfig | AppConfig,
    project_root: Path,
    temp_root: Path,
    *,
    output_stem: str | None = None,
    debug: bool = False,
    formatting: FormattingConfig | None = None,
    language_override: str | None = None,
    job_context: JobRuntimeContext | None = None,
) -> SubtitleGenerationResult | None:
    subtitles_config, formatting_config = _resolve_subtitle_configs(config, formatting)
    if not subtitles_config.enabled:
        LOGGER.info("Phụ đề đã tắt, bỏ qua: %s", video_path)
        return None

    probe = probe_video(video_path)
    if not probe.has_audio:
        LOGGER.warning("Video không có audio, bỏ qua tạo phụ đề: %s", video_path)
        return None

    subtitle_dir = project_root / subtitles_config.output_dir
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    stem = output_stem or video_path.stem
    runtime_context = job_context or _fallback_job_context(video_path, subtitle_dir)
    requested_language = (language_override or subtitles_config.language or "auto").strip() or "auto"

    with job_context_scope(runtime_context):
        manager = TranscriptionManager(subtitles_config, runtime_context)
        resolved_backend = manager.resolve_backend()
        LOGGER.info(
            "%s [TRANSCRIBE] Backend requested: %s | resolved: %s | model: %s | language: %s",
            job_prefix(runtime_context),
            subtitles_config.backend,
            resolved_backend,
            subtitles_config.groq.model if resolved_backend == "groq" else subtitles_config.model_size,
            requested_language,
        )

        try:
            transcription_result = manager.transcribe(video_path, requested_language)
            # use logger to print all transcription_result 
            LOGGER.info("Transcription result: %s", transcription_result)   

            detected_language = transcription_result.language
            raw_segments = [
                {"text": seg.text, "start": seg.start, "end": seg.end, "words": seg.words}
                for seg in transcription_result.segments
            ]
            words: list[SubtitleWord] = [
                SubtitleWord(text=w.text, start=w.start, end=w.end)
                for w in transcription_result.words
            ]
            LOGGER.info(
                "%s [TRANSCRIBE] Backend used: %s | raw_segments=%s | word_timestamps=%s | detected=%s",
                job_prefix(runtime_context),
                transcription_result.backend,
                len(raw_segments),
                len(words),
                detected_language or "unknown",
            )
        except Exception as exc:
            LOGGER.exception("Lỗi khi tạo phụ đề cho %s", video_path)
            LOGGER.error("Chi tiết lỗi phụ đề: %s", exc)
            return None

    with stage_scope(runtime_context, "SPLIT_CAPTION_CUES", logger=LOGGER):
        if _should_fallback_to_segment_cues(raw_segments, words):
            LOGGER.warning(
                "Word timestamps qua it hoac khong du tin cay cho %s, fallback sang segment cues.",
                video_path,
            )
            cues = build_segment_caption_cues(
                raw_segments,
                max_chars_per_line=formatting_config.max_chars_per_line,
                max_lines=formatting_config.max_lines,
            )
        else:
            cues = split_words_into_caption_cues(
                words,
                max_chars_per_line=formatting_config.max_chars_per_line,
                max_lines=formatting_config.max_lines,
                max_chars_per_cue=formatting_config.max_chars_per_cue,
                max_words_per_cue=formatting_config.max_words_per_cue,
                min_duration=formatting_config.min_duration,
                max_duration=formatting_config.max_duration,
                pause_threshold=formatting_config.pause_threshold,
            )
        cues = stabilize_caption_cues(
            cues,
            min_duration=formatting_config.min_duration,
            media_duration=probe.duration,
        )
    if not cues:
        LOGGER.warning("Không có caption cue hợp lệ nào được tạo cho %s", video_path)
        return None

    srt_path: Path | None = None
    vtt_path: Path | None = None
    ass_path: Path | None = None
    if subtitles_config.output_srt or subtitles_config.burn_in:
        with stage_scope(runtime_context, "WRITE_SRT_OR_ASS", logger=LOGGER, format="srt"):
            srt_path = write_srt(cues, subtitle_dir / f"{stem}.srt")
            LOGGER.info("Đã tạo file SRT: %s", srt_path)
    if subtitles_config.output_vtt:
        with stage_scope(runtime_context, "WRITE_SRT_OR_ASS", logger=LOGGER, format="vtt"):
            vtt_path = write_vtt(cues, subtitle_dir / f"{stem}.vtt")
            LOGGER.info("Đã tạo file VTT: %s", vtt_path)
    if subtitles_config.burn_in:
        with stage_scope(runtime_context, "WRITE_SRT_OR_ASS", logger=LOGGER, format="ass"):
            ass_path = write_ass(
                cues,
                subtitle_dir / f"{stem}.ass",
                _style_config_from_formatting(
                    formatting_config,
                    play_res_x=probe.width or 1080,
                    play_res_y=probe.height or 1920,
                ),
            )
            LOGGER.info("Đã tạo file ASS cho burn-in: %s", ass_path)

    max_lines_found = max((len(cue.lines) for cue in cues), default=0)
    LOGGER.info(
        "Tạo phụ đề hoàn tất cho %s | backend=%s | language=%s | detected=%s | raw_segments=%s | word_timestamps=%s | cues=%s | max_lines=%s",
        video_path,
        transcription_result.backend,
        requested_language,
        detected_language or "unknown",
        len(raw_segments),
        len(words),
        len(cues),
        max_lines_found,
    )
    return SubtitleGenerationResult(
        media_path=video_path,
        subtitle_dir=subtitle_dir,
        srt_path=srt_path,
        vtt_path=vtt_path,
        ass_path=ass_path,
        cues=cues,
        language=requested_language,
        detected_language=detected_language,
    )


def burn_subtitles_into_video(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    config: AppConfig,
    *,
    project_root: Path | None = None,
    temp_root: Path | None = None,
    encoder_selection: EncoderSelection | None = None,
    debug: bool = False,
    job_context: JobRuntimeContext | None = None,
    subtitle_result: SubtitleGenerationResult | None = None,
) -> Path:
    probe = probe_video(video_path)
    available_encoders = detect_available_encoders()
    encoder = encoder_selection or select_encoder(
        config.encoder,
        available_encoders,
        width=probe.width or 1280,
        height=probe.height or 720,
        allow_cpu_fallback=config.encoder.allow_cpu_fallback,
        smoke_test_on_startup=config.encoder.smoke_test_on_startup,
        cache_capability_results=config.encoder.cache_capability_results,
        container_gpu_mode=config.runtime.container_gpu_mode if config.runtime.prefer_native_hardware_acceleration else "cpu",
        vaapi_device=config.vaapi.device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_filter_chain = _build_burn_filter(subtitle_path, config)
    runtime_context = job_context or _fallback_job_context(video_path, output_path.parent)
    renderer = (config.formatting.caption_renderer or "rounded_box").strip().lower()
    LOGGER.info(
        (
            "Burn phụ đề | input=%s | subtitle=%s | output=%s | "
            "video=%sx%s | dar=%s | sar=%s | time_base=%s | "
        "audio_codec=%s | audio_sr=%s | audio_ch=%s | audio_dur=%.3f | legacy_filter=%s"
        ),
        video_path,
        subtitle_path,
        output_path,
        probe.width,
        probe.height,
        probe.display_aspect_ratio or _ratio_string(probe.width, probe.height),
        probe.sample_aspect_ratio or "unknown",
        probe.time_base or "unknown",
        probe.audio_codec or "none",
        probe.audio_sample_rate or 0,
        probe.audio_channels or 0,
        probe.audio_duration or 0.0,
        legacy_filter_chain,
    )
    LOGGER.info(
        "%s [CAPTION_STYLE] renderer=%s font_file=%s font_name=%s font_size=%s background=%s opacity=%s",
        job_prefix(runtime_context),
        renderer,
        config.formatting.caption_font_file,
        config.formatting.caption_font_name,
        config.formatting.caption_font_size,
        config.formatting.caption_background_color,
        config.formatting.caption_background_opacity,
    )

    if renderer == "rounded_box":
        if project_root is None or temp_root is None:
            LOGGER.warning(
                "%s [BURN_CAPTIONS] Thiếu project_root/temp_root, fallback sang ASS renderer.",
                job_prefix(runtime_context),
            )
        elif subtitle_result is None or not subtitle_result.cues:
            LOGGER.warning(
                "%s [BURN_CAPTIONS] Không có cues đã split để render rounded box, fallback sang ASS renderer.",
                job_prefix(runtime_context),
            )
        else:
            from processing.caption_renderer import burn_rounded_captions

            try:
                return burn_rounded_captions(
                    video_path,
                    output_path,
                    subtitle_result.cues,
                    config,
                    project_root=project_root,
                    temp_root=temp_root,
                    encoder_selection=encoder,
                    debug=debug,
                    job_context=runtime_context,
                )
            except Exception as exc:
                LOGGER.exception("%s [BURN_CAPTIONS] Rounded-box renderer thất bại", job_prefix(runtime_context))
                LOGGER.warning(
                    "%s [BURN_CAPTIONS] Fallback sang ASS renderer do lỗi: %s",
                    job_prefix(runtime_context),
                    exc,
                )
    LOGGER.info(
        "%s [BURN_CAPTIONS] Input path=%s | subtitle path=%s | output path=%s | encoder=%s | audio_mode=copy->aac",
        job_prefix(runtime_context),
        video_path,
        subtitle_path,
        output_path,
        encoder.codec_name,
    )

    filter_chain = _vaapi_aware_filter_chain(legacy_filter_chain, encoder.backend)
    args = _build_burn_command(
        video_path=video_path,
        output_path=output_path,
        config=config,
        encoder=encoder,
        filter_chain=filter_chain,
        has_audio=probe.has_audio,
        audio_mode="copy",
    )
    LOGGER.info("Burn FFmpeg command (audio=copy): %s", " ".join(args))

    try:
        with stage_scope(runtime_context, "BURN_CAPTIONS", logger=LOGGER, audio_mode="copy", start_level=logging.INFO):
            run_command(args, debug=debug)
        audio_mode = "copy"
    except FFmpegError as exc:
        if encoder.backend.startswith("cpu") or not is_hardware_runtime_failure(exc.result.stderr):
            if not probe.has_audio:
                raise
            LOGGER.warning("Copy audio thất bại khi burn captions, thử AAC fallback an toàn.")
            args = _build_burn_command(
                video_path=video_path,
                output_path=output_path,
                config=config,
                encoder=encoder,
                filter_chain=filter_chain,
                has_audio=probe.has_audio,
                audio_mode="aac",
            )
            LOGGER.info("Burn FFmpeg command (audio=aac): %s", " ".join(args))
            with stage_scope(runtime_context, "BURN_CAPTIONS", logger=LOGGER, audio_mode="aac", start_level=logging.INFO):
                run_command(args, debug=debug)
            audio_mode = "aac"
        else:
            reason = extract_hardware_failure_reason(exc.result.stderr)
            LOGGER.warning(
                "%s [BURN_CAPTIONS] Hardware backend %s failed: %s | retrying with CPU",
                job_prefix(runtime_context),
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
            cpu_filter_chain = _vaapi_aware_filter_chain(legacy_filter_chain, cpu_encoder.backend)
            args = _build_burn_command(
                video_path=video_path,
                output_path=output_path,
                config=config,
                encoder=cpu_encoder,
                filter_chain=cpu_filter_chain,
                has_audio=probe.has_audio,
                audio_mode="copy",
            )
            LOGGER.debug("%s [BURN_CAPTIONS] ffmpeg_args_cpu_fallback=%s", job_prefix(runtime_context), " ".join(args))
            try:
                with stage_scope(runtime_context, "BURN_CAPTIONS", logger=LOGGER, audio_mode="copy", start_level=logging.INFO):
                    run_command(args, debug=debug)
                audio_mode = "copy"
            except FFmpegError:
                if not probe.has_audio:
                    raise
                LOGGER.warning("Copy audio thất bại khi burn captions, thử AAC fallback an toàn.")
                args = _build_burn_command(
                    video_path=video_path,
                    output_path=output_path,
                    config=config,
                    encoder=cpu_encoder,
                    filter_chain=cpu_filter_chain,
                    has_audio=probe.has_audio,
                    audio_mode="aac",
                )
                LOGGER.info("Burn FFmpeg command (audio=aac): %s", " ".join(args))
                with stage_scope(runtime_context, "BURN_CAPTIONS", logger=LOGGER, audio_mode="aac", start_level=logging.INFO):
                    run_command(args, debug=debug)
                audio_mode = "aac"

    burned_probe = probe_video(output_path)
    _log_probe_comparison(probe, burned_probe, runtime_context, stage_name="VALIDATE_BURNED_OUTPUT")
    LOGGER.info(
        (
            "Burn output | path=%s | video=%sx%s | dar=%s | sar=%s | time_base=%s | "
            "audio_codec=%s | audio_sr=%s | audio_ch=%s | audio_dur=%.3f | audio_mode=%s"
        ),
        output_path,
        burned_probe.width,
        burned_probe.height,
        burned_probe.display_aspect_ratio or _ratio_string(burned_probe.width, burned_probe.height),
        burned_probe.sample_aspect_ratio or "unknown",
        burned_probe.time_base or "unknown",
        burned_probe.audio_codec or "none",
        burned_probe.audio_sample_rate or 0,
        burned_probe.audio_channels or 0,
        burned_probe.audio_duration or 0.0,
        audio_mode,
    )
    return output_path


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def format_vtt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def format_ass_timestamp(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(total_cs, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02}:{secs:02}.{centiseconds:02}"


# --- Helper functions (kept for internal use by other modules) ---

def _should_break_before_add(
    *,
    current: list[SubtitleWord],
    pause: float,
    tentative_text: str,
    current_duration: float,
    max_chars_per_line: int,
    max_lines: int,
    max_chars_per_cue: int,
    max_duration: float,
    pause_threshold: float,
) -> bool:
    if pause > pause_threshold:
        return True
    if current_duration >= max_duration:
        return True
    if _text_length(tentative_text) > max_chars_per_cue:
        return True
    if _would_exceed_caption_lines(tentative_text, max_chars_per_line, max_lines):
        return True
    return False


def _build_caption_cue(
    words: list[SubtitleWord],
    *,
    max_chars_per_line: int,
    max_lines: int,
) -> SubtitleCue:
    start = words[0].start
    end = max(words[-1].end, start + 0.01)
    text = _join_words(words)
    lines = wrap_caption_text(text, max_chars_per_line=max_chars_per_line, max_lines=max_lines)
    if not lines:
        lines = [text]
    return SubtitleCue(start=start, end=end, text="\n".join(lines), lines=lines)


def build_segment_caption_cues(
    segments: Iterable[dict[str, Any] | Any],
    *,
    max_chars_per_line: int,
    max_lines: int,
) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for item in segments:
        text = ""
        start = 0.0
        end = 0.0
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").replace("\n", " ").strip()
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
        else:
            text = str(getattr(item, "text", "") or "").replace("\n", " ").strip()
            start = float(getattr(item, "start", 0.0) or 0.0)
            end = float(getattr(item, "end", start) or start)
        if not text:
            continue
        lines = wrap_caption_text(text, max_chars_per_line=max_chars_per_line, max_lines=max_lines)
        if not lines:
            lines = [text]
        cues.append(
            SubtitleCue(
                start=start,
                end=max(end, start + 0.01),
                text="\n".join(lines),
                lines=lines,
            )
        )
    return cues


def stabilize_caption_cues(
    cues: Iterable[SubtitleCue],
    *,
    min_duration: float,
    media_duration: float | None = None,
) -> list[SubtitleCue]:
    ordered = sorted(
        (
            SubtitleCue(
                start=max(0.0, float(cue.start)),
                end=max(float(cue.end), float(cue.start)),
                text=cue.text,
                lines=list(cue.lines),
            )
            for cue in cues
            if cue.text.strip()
        ),
        key=lambda cue: (cue.start, cue.end),
    )
    if not ordered:
        return []

    bounded_duration = max(0.0, float(media_duration or 0.0))
    stabilized: list[SubtitleCue] = []
    for index, cue in enumerate(ordered):
        next_start = ordered[index + 1].start if index + 1 < len(ordered) else None
        desired_end = max(cue.end, cue.start + max(0.0, float(min_duration)))
        if next_start is not None and next_start > cue.end:
            desired_end = min(desired_end, next_start)
        if bounded_duration > 0:
            desired_end = min(desired_end, bounded_duration)
        stabilized.append(
            SubtitleCue(
                start=cue.start,
                end=max(cue.end, desired_end),
                text=cue.text,
                lines=list(cue.lines),
            )
        )
    return stabilized


def _should_fallback_to_segment_cues(
    raw_segments: list[dict[str, Any]],
    words: list[SubtitleWord],
) -> bool:
    if not raw_segments:
        return False
    if not words:
        return True
    segment_token_count = sum(len(str(seg.get("text", "") or "").split()) for seg in raw_segments)
    if segment_token_count <= 0:
        return False
    return len(words) <= 1 and segment_token_count > len(words)


def _join_words(words: list[SubtitleWord]) -> str:
    parts: list[str] = []
    for word in words:
        token = str(word.text).replace("\n", " ").strip()
        if not token:
            continue
        parts.append(token)
    return " ".join(parts).strip()


def _entry_lines(entry: SubtitleCue) -> list[str]:
    lines = [line for line in (entry.lines or entry.text.splitlines()) if line.strip()]
    return lines or [entry.text]


def _entry_text(entry: SubtitleCue) -> str:
    return "\n".join(_entry_lines(entry))


def _text_length(text: str) -> int:
    return len(text.replace("\n", " ").strip())


def _ends_caption(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped.endswith(_PREFERRED_PUNCTUATION)


def _wrap_caption_text(
    text: str,
    max_chars_per_line: int,
    max_lines: int,
) -> tuple[list[str], bool]:
    words = text.split()
    if not words:
        return [], False

    lines: list[str] = []
    current: list[str] = []
    overflow = False

    for word in words:
        projected = len(" ".join(current)) + len(word) + (1 if current else 0)
        if current and projected > max_chars_per_line:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) == max_lines:
                overflow = True
                break
        else:
            current.append(word)

    if not overflow and current:
        if len(lines) < max_lines:
            lines.append(" ".join(current))
        else:
            overflow = True
    if len(lines) > max_lines:
        overflow = True
        lines = lines[:max_lines]
    return lines, overflow


def _would_exceed_caption_lines(text: str, max_chars_per_line: int, max_lines: int) -> bool:
    _, overflow = _wrap_caption_text(text, max_chars_per_line, max_lines)
    return overflow


def _coerce_word(item: SubtitleWord | dict[str, Any] | Any) -> SubtitleWord | None:
    if isinstance(item, SubtitleWord):
        return item
    text = getattr(item, "text", None)
    if text is None:
        text = getattr(item, "word", None)
    if text is None and isinstance(item, dict):
        text = item.get("text") or item.get("word")
    if text is None:
        return None
    start = getattr(item, "start", None)
    end = getattr(item, "end", None)
    if start is None and isinstance(item, dict):
        start = item.get("start", 0.0)
    if end is None and isinstance(item, dict):
        end = item.get("end", start if start is not None else 0.0)
    start_value = float(start if start is not None else 0.0)
    end_value = float(end if end is not None else start_value)
    if end_value < start_value:
        end_value = start_value
    return SubtitleWord(text=str(text).replace("\n", " ").strip(), start=start_value, end=end_value)


def _resolve_subtitle_configs(
    config: SubtitlesConfig | AppConfig,
    formatting: FormattingConfig | None,
) -> tuple[SubtitlesConfig, FormattingConfig]:
    if isinstance(config, AppConfig):
        return config.subtitles, formatting or config.formatting
    return config, formatting or FormattingConfig()


def _style_config_from_formatting(
    formatting: FormattingConfig,
    *,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
) -> AssStyleConfig:
    return AssStyleConfig(
        font_name=formatting.caption_font_name,
        font_size=formatting.caption_font_size,
        outline=formatting.caption_outline,
        shadow=formatting.caption_shadow,
        margin_v=max(0, formatting.caption_margin_v + formatting.caption_vertical_offset),
        alignment=formatting.caption_position,
        play_res_x=max(2, int(play_res_x)),
        play_res_y=max(2, int(play_res_y)),
        text_color=formatting.caption_text_color,
        outline_color=formatting.caption_outline_color,
        background_color=formatting.caption_background_color,
        text_opacity=formatting.caption_text_opacity,
        outline_opacity=formatting.caption_outline_opacity,
        background_opacity=formatting.caption_background_opacity,
        box_enabled=formatting.caption_box_enabled,
    )


def _resolve_ass_alignment(position: str) -> int:
    normalized = position.lower().strip()
    if normalized == "top":
        return 8
    if normalized == "middle":
        return 5
    return 2


def _build_burn_filter(subtitle_path: Path, config: AppConfig) -> str:
    if subtitle_path.suffix.lower() == ".ass":
        return build_ass_burn_filter(subtitle_path)
    return build_subtitles_burn_filter(
        subtitle_path,
        font_size=config.formatting.caption_font_size,
        margin_v=config.formatting.caption_margin_v,
    )


def _vaapi_aware_filter_chain(filter_chain: str, encoder_backend: str) -> str:
    if encoder_backend.startswith("vaapi"):
        return f"{filter_chain},format=nv12,hwupload"
    return filter_chain


def _build_burn_command(
    *,
    video_path: Path,
    output_path: Path,
    config: AppConfig,
    encoder: EncoderSelection,
    filter_chain: str,
    has_audio: bool,
    audio_mode: str,
) -> list[str]:
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vf",
        filter_chain,
        "-fps_mode",
        "passthrough",
    ]
    if has_audio:
        args.extend(["-map", "0:a?"])
    args.extend(encoder.args)
    args.extend(["-pix_fmt", config.encoder.pix_fmt])
    if has_audio:
        if audio_mode == "copy":
            args.extend(["-c:a", "copy"])
        else:
            args.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000"])
    else:
        args.append("-an")
    if config.encoder.faststart:
        args.extend(["-movflags", "+faststart"])
    args.extend(["-map_metadata", "0", "-map_chapters", "0"])
    args.append(str(output_path))
    return args


def _log_probe_comparison(
    before: Any,
    after: Any,
    context: JobRuntimeContext,
    *,
    stage_name: str,
) -> None:
    LOGGER.info(
        "%s [%s] output before=%sx%s after=%sx%s dar_before=%s dar_after=%s sar_before=%s sar_after=%s duration_before=%.3f duration_after=%.3f video_codec_before=%s video_codec_after=%s audio_codec_before=%s audio_codec_after=%s audio_sr_before=%s audio_sr_after=%s audio_ch_before=%s audio_ch_after=%s",
        job_prefix(context),
        stage_name,
        before.width,
        before.height,
        after.width,
        after.height,
        before.display_aspect_ratio or _ratio_string(before.width, before.height),
        after.display_aspect_ratio or _ratio_string(after.width, after.height),
        before.sample_aspect_ratio or "unknown",
        after.sample_aspect_ratio or "unknown",
        before.duration,
        after.duration,
        before.video_codec or "unknown",
        after.video_codec or "unknown",
        before.audio_codec or "none",
        after.audio_codec or "none",
        before.audio_sample_rate or 0,
        after.audio_sample_rate or 0,
        before.audio_channels or 0,
        after.audio_channels or 0,
    )
    if (before.width, before.height) != (after.width, after.height):
        LOGGER.warning(
            "%s [%s] Resolution changed unexpectedly | before=%sx%s after=%sx%s",
            job_prefix(context),
            stage_name,
            before.width,
            before.height,
            after.width,
            after.height,
        )
    if (before.display_aspect_ratio or "") != (after.display_aspect_ratio or ""):
        LOGGER.warning(
            "%s [%s] Aspect ratio changed unexpectedly | before=%s after=%s",
            job_prefix(context),
            stage_name,
            before.display_aspect_ratio or _ratio_string(before.width, before.height),
            after.display_aspect_ratio or _ratio_string(after.width, after.height),
        )
    if abs((before.duration or 0.0) - (after.duration or 0.0)) > 0.5:
        LOGGER.warning(
            "%s [%s] Duration changed unexpectedly | before=%.3f after=%.3f",
            job_prefix(context),
            stage_name,
            before.duration,
            after.duration,
        )
    if (before.audio_sample_rate or 0) != (after.audio_sample_rate or 0):
        LOGGER.warning(
            "%s [%s] Audio sample rate changed unexpectedly | before=%s after=%s",
            job_prefix(context),
            stage_name,
            before.audio_sample_rate or 0,
            after.audio_sample_rate or 0,
        )
    if (before.audio_channels or 0) != (after.audio_channels or 0):
        LOGGER.warning(
            "%s [%s] Audio channel count changed unexpectedly | before=%s after=%s",
            job_prefix(context),
            stage_name,
            before.audio_channels or 0,
            after.audio_channels or 0,
        )


def _ratio_string(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    return f"{width}:{height}"


def _ass_colour(hex_colour: str, opacity: float) -> str:
    r, g, b = _parse_hex_colour(hex_colour)
    alpha = _ass_alpha(opacity)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _parse_hex_colour(value: str) -> tuple[int, int, int]:
    raw = value.strip().lstrip("#")
    if len(raw) == 6:
        pass
    elif len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    else:
        return 255, 255, 255
    try:
        r = int(raw[0:2], 16)
        g = int(raw[2:4], 16)
        b = int(raw[4:6], 16)
    except ValueError:
        return 255, 255, 255
    return r, g, b


def _ass_alpha(opacity: float) -> int:
    clamped = max(0.0, min(1.0, float(opacity)))
    return int(round((1.0 - clamped) * 255))


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r", "")
    )


def _fallback_job_context(video_path: Path, subtitle_dir: Path) -> JobRuntimeContext:
    return JobRuntimeContext(
        job_id=f"subtitle-{abs(hash(video_path.as_posix())):x}"[:12],
        source="local_input",
        input_path=video_path,
        output_path=subtitle_dir,
        worker_slot=None,
        worker_total=None,
        thread_name="subtitle-fallback",
        pid=0,
    )
