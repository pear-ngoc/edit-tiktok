from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

from ffmpeg_tools.encoders import detect_available_encoders, select_encoder
from ffmpeg_tools.probe import probe_video
from ffmpeg_tools.runner import FFmpegError, run_command
from models import AppConfig, EncoderSelection, FormattingConfig
from processing.subtitles import SubtitleCue, wrap_caption_text
from utils.paths import resolve_project_path
from utils.runtime_logging import (
    JobRuntimeContext,
    job_prefix,
    stage_scope,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedCaptionStyle:
    renderer: str
    font_name: str
    font_file: Path | None
    font_fallback: str
    font_size: int
    scaled_font_size: int
    text_color: tuple[int, int, int, int]
    background_color: tuple[int, int, int, int]
    outline_color: tuple[int, int, int, int]
    shadow_color: tuple[int, int, int, int]
    padding_x: int
    padding_y: int
    border_radius: int
    margin_v: int
    vertical_offset: int
    max_width_percent: int
    max_lines: int
    box_enabled: bool
    shadow_enabled: bool
    shadow_offset_x: int
    shadow_offset_y: int
    shadow_blur: int
    outline: int
    video_width: int
    video_height: int
    scale_factor: float

    @property
    def max_text_width_px(self) -> int:
        return max(1, int(self.video_width * self.max_width_percent / 100))


@dataclass(slots=True)
class RenderedCaptionAsset:
    image_path: Path
    start: float
    end: float


@dataclass(slots=True)
class CaptionFontResolution:
    font_path: Path | None
    font_name: str
    warning: str | None = None


def available_caption_fonts(project_root: Path) -> list[Path]:
    font_dir = resolve_project_path(project_root, "assets/font")
    if not font_dir.exists():
        return []
    return sorted(
        [
            path
            for path in font_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}
        ],
        key=lambda item: item.name.lower(),
    )


def resolve_caption_font(project_root: Path, formatting: FormattingConfig) -> CaptionFontResolution:
    font_dir = resolve_project_path(project_root, "assets/font").resolve()
    raw = (formatting.caption_font_file or "").strip()
    fallback = (formatting.caption_font_fallback or "Arial").strip() or "Arial"
    if not raw:
        return CaptionFontResolution(font_path=None, font_name=fallback, warning="Thiếu caption_font_file, dùng font dự phòng.")

    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve()
            if font_dir not in candidate.parents and candidate != font_dir:
                return CaptionFontResolution(
                    font_path=None,
                    font_name=fallback,
                    warning=f"Bỏ qua font ngoài assets/font: {raw}",
                )
        except Exception:
            return CaptionFontResolution(
                font_path=None,
                font_name=fallback,
                warning=f"Không thể xác thực font: {raw}",
            )
    else:
        candidate = (font_dir / candidate).resolve()

    try:
        candidate.relative_to(font_dir)
    except Exception:
        return CaptionFontResolution(
            font_path=None,
            font_name=fallback,
            warning=f"Bỏ qua font ngoài assets/font: {raw}",
        )

    if not candidate.exists() or candidate.suffix.lower() not in {".ttf", ".otf"}:
        return CaptionFontResolution(
            font_path=None,
            font_name=fallback,
            warning=f"Không tìm thấy font {raw}, dùng font dự phòng.",
        )

    return CaptionFontResolution(font_path=candidate, font_name=candidate.name)


def resolve_caption_style(
    formatting: FormattingConfig,
    *,
    video_width: int,
    video_height: int,
    project_root: Path,
) -> tuple[ResolvedCaptionStyle, CaptionFontResolution]:
    scale_factor = _caption_scale_factor(video_width, video_height)
    font_resolution = resolve_caption_font(project_root, formatting)
    style = ResolvedCaptionStyle(
        renderer=(formatting.caption_renderer or "rounded_box").strip().lower(),
        font_name=font_resolution.font_path.name if font_resolution.font_path else (formatting.caption_font_name or font_resolution.font_name),
        font_file=font_resolution.font_path,
        font_fallback=formatting.caption_font_fallback or "Arial",
        font_size=max(8, int(formatting.caption_font_size)),
        scaled_font_size=max(8, int(round(formatting.caption_font_size * scale_factor))),
        text_color=_rgba_from_hex(formatting.caption_text_color, formatting.caption_text_opacity),
        background_color=_rgba_from_hex(
            formatting.caption_background_color,
            formatting.caption_background_opacity,
        ),
        outline_color=_rgba_from_hex(formatting.caption_outline_color, formatting.caption_outline_opacity),
        shadow_color=_rgba_from_hex(formatting.caption_shadow_color, formatting.caption_shadow_opacity),
        padding_x=max(0, int(round(formatting.caption_padding_x * scale_factor))),
        padding_y=max(0, int(round(formatting.caption_padding_y * scale_factor))),
        border_radius=max(0, int(round(formatting.caption_border_radius * scale_factor))),
        margin_v=max(0, int(round(formatting.caption_margin_v * scale_factor))),
        vertical_offset=int(round(formatting.caption_vertical_offset * scale_factor)),
        max_width_percent=max(10, min(95, int(formatting.caption_max_width_percent))),
        max_lines=max(1, int(formatting.max_lines)),
        box_enabled=bool(formatting.caption_box_enabled),
        shadow_enabled=bool(formatting.caption_shadow_enabled),
        shadow_offset_x=int(round(formatting.caption_shadow_offset_x * scale_factor)),
        shadow_offset_y=int(round(formatting.caption_shadow_offset_y * scale_factor)),
        shadow_blur=max(0, int(round(formatting.caption_shadow_blur * scale_factor))),
        outline=max(0, int(round(formatting.caption_outline * scale_factor))),
        video_width=max(2, int(video_width)),
        video_height=max(2, int(video_height)),
        scale_factor=scale_factor,
    )
    return style, font_resolution


def measure_caption_text(
    lines: list[str],
    *,
    font: Any,
    stroke_width: int = 0,
    line_spacing: int = 4,
) -> tuple[int, int, list[tuple[int, int]]]:
    image = _create_measure_image()
    draw = _image_draw(image)
    line_sizes: list[tuple[int, int]] = []
    widths: list[int] = []
    heights: list[int] = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font, stroke_width=stroke_width)
        width = max(1, int(bbox[2] - bbox[0]))
        height = max(1, int(bbox[3] - bbox[1]))
        line_sizes.append((width, height))
        widths.append(width)
        heights.append(height)
    total_height = sum(heights) + max(0, len(lines) - 1) * line_spacing
    return max(widths or [1]), max(1, total_height), line_sizes


def render_caption_image(
    cue: SubtitleCue,
    style: ResolvedCaptionStyle,
    output_path: Path,
    *,
    warn_on_fallback: bool = True,
) -> Path:
    if not cue.lines:
        cue_lines = wrap_caption_text(
            cue.text,
            max_chars_per_line=20,
            max_lines=style.max_lines,
        )
    else:
        cue_lines = list(cue.lines[: style.max_lines])
    cue_lines = [line for line in cue_lines if line.strip()]
    if not cue_lines:
        cue_lines = [cue.text.strip() or " "]

    image_module, draw_module, image_filter, font_module = _import_pillow()
    font, used_fallback = _load_font(font_module, style, style.scaled_font_size, warn_on_fallback=warn_on_fallback)
    if used_fallback and warn_on_fallback:
        LOGGER.warning("Dùng font dự phòng cho caption: %s", style.font_fallback)

    font_size = style.scaled_font_size
    stroke_width = style.outline if style.outline_color[3] > 0 and style.outline > 0 else 0
    line_spacing = max(2, int(round(font_size * 0.16)))
    max_text_width = max(1, style.max_text_width_px - max(0, style.padding_x * 2))

    for _ in range(8):
        max_line_width, text_height, line_sizes = measure_caption_text(
            cue_lines,
            font=font,
            stroke_width=stroke_width,
            line_spacing=line_spacing,
        )
        if max_line_width <= max_text_width:
            break
        shrink_ratio = max_text_width / max(1, max_line_width)
        next_size = max(18, int(font_size * shrink_ratio))
        if next_size >= font_size:
            break
        font_size = next_size
        font, _ = _load_font(font_module, style, font_size, warn_on_fallback=warn_on_fallback)
        line_spacing = max(2, int(round(font_size * 0.16)))
    else:
        max_line_width, text_height, line_sizes = measure_caption_text(
            cue_lines,
            font=font,
            stroke_width=stroke_width,
            line_spacing=line_spacing,
        )

    box_width = min(style.max_text_width_px, max_line_width + style.padding_x * 2)
    box_height = text_height + style.padding_y * 2

    shadow_pad = 0
    if style.box_enabled and style.shadow_enabled and style.shadow_color[3] > 0:
        shadow_pad = max(0, style.shadow_blur * 2 + max(abs(style.shadow_offset_x), abs(style.shadow_offset_y)) + 4)
    else:
        shadow_pad = max(2, style.padding_x // 2)

    canvas_width = box_width + shadow_pad * 2
    canvas_height = box_height + shadow_pad * 2
    canvas = image_module.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    if style.box_enabled and style.shadow_enabled and style.shadow_color[3] > 0:
        shadow_layer = image_module.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        shadow_draw = draw_module.Draw(shadow_layer)
        shadow_box = (
            shadow_pad + style.shadow_offset_x,
            shadow_pad + style.shadow_offset_y,
            shadow_pad + style.shadow_offset_x + box_width,
            shadow_pad + style.shadow_offset_y + box_height,
        )
        shadow_draw.rounded_rectangle(
            shadow_box,
            radius=max(0, style.border_radius),
            fill=style.shadow_color,
        )
        if style.shadow_blur > 0:
            shadow_layer = shadow_layer.filter(image_filter.GaussianBlur(radius=style.shadow_blur))
        canvas = image_module.alpha_composite(canvas, shadow_layer)

    box_x = shadow_pad
    box_y = shadow_pad
    if style.box_enabled:
        draw = draw_module.Draw(canvas)
        box_fill = style.background_color
        draw.rounded_rectangle(
            (box_x, box_y, box_x + box_width, box_y + box_height),
            radius=max(0, style.border_radius),
            fill=box_fill,
        )

    draw = draw_module.Draw(canvas)
    text_block_y = box_y + style.padding_y + max(0, (box_height - style.padding_y * 2 - text_height) // 2)
    current_y = text_block_y
    for index, line in enumerate(cue_lines):
        line_width, line_height = line_sizes[index]
        text_x = box_x + max(0, (box_width - line_width) // 2)
        stroke_fill = style.outline_color if stroke_width > 0 else None
        draw.text(
            (text_x, current_y),
            line,
            font=font,
            fill=style.text_color,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        current_y += line_height + line_spacing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def wrap_caption_text(
    text: str,
    max_chars_per_line: int = 20,
    max_lines: int = 2,
) -> list[str]:
    from processing.subtitles import wrap_caption_text as _wrap_caption_text

    return _wrap_caption_text(
        text,
        max_chars_per_line=max_chars_per_line,
        max_lines=max_lines,
    )


def build_caption_overlay_filter(
    caption_assets: list[RenderedCaptionAsset],
    *,
    margin_v: int,
    vertical_offset: int = 0,
) -> tuple[list[str], str, str]:
    input_args: list[str] = []
    filter_parts: list[str] = []
    current_label = "0:v"
    total_offset = max(0, int(margin_v + vertical_offset))
    for index, asset in enumerate(caption_assets, start=1):
        input_args.extend(["-loop", "1", "-i", str(asset.image_path)])
        cap_label = f"cap{index}"
        out_label = f"v{index}"
        filter_parts.append(f"[{index}:v]format=rgba[{cap_label}]")
        filter_parts.append(
            f"[{current_label}][{cap_label}]overlay=x=(W-w)/2:y=H-h-{total_offset}:"
            f"enable='between(t,{asset.start:.3f},{asset.end:.3f})':shortest=1[{out_label}]"
        )
        current_label = out_label
    final_label = f"[{current_label}]"
    return input_args, ";".join(filter_parts), final_label


def burn_rounded_captions(
    video_path: Path,
    output_path: Path,
    cues: Iterable[SubtitleCue],
    config: AppConfig,
    *,
    project_root: Path,
    temp_root: Path,
    encoder_selection: EncoderSelection | None = None,
    debug: bool = False,
    job_context: JobRuntimeContext | None = None,
) -> Path:
    probe = probe_video(video_path)
    available_encoders = detect_available_encoders()
    encoder = encoder_selection or select_encoder(
        config.encoder,
        available_encoders,
        width=probe.width or 1280,
        height=probe.height or 720,
    )
    runtime_context = job_context or _fallback_job_context(video_path, output_path.parent)
    resolved_style, font_resolution = resolve_caption_style(
        config.formatting,
        video_width=probe.width or 1080,
        video_height=probe.height or 1920,
        project_root=project_root,
    )
    _log_caption_style(runtime_context, resolved_style, font_resolution)

    captions = [cue for cue in cues if cue.text.strip() and cue.end > cue.start]
    if not captions:
        raise RuntimeError("Không có caption cue hợp lệ để burn")

    job_temp_root = (temp_root if temp_root is not None else output_path.parent / "temp") / runtime_context.job_id
    caption_dir = job_temp_root / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)

    rendered_assets: list[RenderedCaptionAsset] = []
    caption_mode = (config.logging.caption_log_mode or "summary").strip().lower()
    caption_started = time.monotonic()
    try:
        if caption_mode != "none":
            LOGGER.info(
                "%s CAPTIONS Rendering %s images | Pillow CPU",
                job_prefix(runtime_context),
                len(captions),
            )
        with stage_scope(runtime_context, "BUILD_CAPTION_IMAGES", logger=LOGGER, count=len(captions), start_level=logging.DEBUG):
            for index, cue in enumerate(captions, start=1):
                image_path = caption_dir / f"cue_{index:04d}.png"
                with stage_scope(
                    runtime_context,
                    f"CAPTION_IMAGE {index}/{len(captions)}",
                    logger=LOGGER,
                    start=f"{cue.start:.3f}",
                    end=f"{cue.end:.3f}",
                ):
                    render_caption_image(cue, resolved_style, image_path, warn_on_fallback=False)
                rendered_assets.append(RenderedCaptionAsset(image_path=image_path, start=cue.start, end=cue.end))
        if caption_mode != "none":
            LOGGER.info(
                "%s CAPTIONS Done | %s images | %.2fs",
                job_prefix(runtime_context),
                len(captions),
                time.monotonic() - caption_started,
            )

        input_args, filter_complex, final_label = build_caption_overlay_filter(
            rendered_assets,
            margin_v=resolved_style.margin_v,
            vertical_offset=resolved_style.vertical_offset,
        )
        args = _build_burn_command(
            video_path=video_path,
            output_path=output_path,
            config=config,
            encoder=encoder,
            filter_chain=filter_complex,
            has_audio=probe.has_audio,
            audio_mode="copy",
            overlay_input_args=input_args,
            video_map=final_label,
        )
        LOGGER.info(
            "%s [BURN_CAPTIONS] Renderer=%s | font=%s | font_file=%s | box=%s | size=%s | padding=(%s,%s) | radius=%s | shadow=%s | max_width=%s%%",
            job_prefix(runtime_context),
            resolved_style.renderer,
            resolved_style.font_name,
            resolved_style.font_file.name if resolved_style.font_file else resolved_style.font_fallback,
            "on" if resolved_style.box_enabled else "off",
            resolved_style.scaled_font_size,
            resolved_style.padding_x,
            resolved_style.padding_y,
            resolved_style.border_radius,
            "on" if resolved_style.shadow_enabled else "off",
            resolved_style.max_width_percent,
        )
        LOGGER.debug("%s [BURN_CAPTIONS] filter_complex=%s", job_prefix(runtime_context), filter_complex)
        LOGGER.debug("%s [BURN_CAPTIONS] ffmpeg_args=%s", job_prefix(runtime_context), " ".join(args))
        with stage_scope(runtime_context, "BURN_CAPTIONS", logger=LOGGER, renderer=resolved_style.renderer):
            try:
                run_command(args, debug=debug, stderr_tail_lines=config.logging.ffmpeg_stderr_tail_lines)
            except FFmpegError:
                if not probe.has_audio:
                    raise
                LOGGER.warning("Audio copy thất bại khi burn rounded captions, thử AAC fallback an toàn.")
                args = _build_burn_command(
                    video_path=video_path,
                    output_path=output_path,
                    config=config,
                    encoder=encoder,
                    filter_chain=filter_complex,
                    has_audio=probe.has_audio,
                    audio_mode="aac",
                    overlay_input_args=input_args,
                    video_map=final_label,
                )
                LOGGER.debug("%s [BURN_CAPTIONS] ffmpeg_args_fallback=%s", job_prefix(runtime_context), " ".join(args))
                run_command(args, debug=debug, stderr_tail_lines=config.logging.ffmpeg_stderr_tail_lines)
    except Exception:
        if not config.logging.retain_failed_temp:
            shutil.rmtree(job_temp_root, ignore_errors=True)
        raise
    else:
        shutil.rmtree(job_temp_root, ignore_errors=True)

    burned_probe = probe_video(output_path)
    LOGGER.info(
        "%s [VALIDATE_BURNED_OUTPUT] width=%sx%s dar=%s audio_codec=%s sample_rate=%s channels=%s duration=%.3f",
        job_prefix(runtime_context),
        burned_probe.width,
        burned_probe.height,
        burned_probe.display_aspect_ratio or f"{burned_probe.width}:{burned_probe.height}",
        burned_probe.audio_codec or "none",
        burned_probe.audio_sample_rate or 0,
        burned_probe.audio_channels or 0,
        burned_probe.duration,
    )
    return output_path


def list_caption_font_names(project_root: Path) -> list[str]:
    return [path.name for path in available_caption_fonts(project_root)]


def _build_burn_command(
    *,
    video_path: Path,
    output_path: Path,
    config: AppConfig,
    encoder: EncoderSelection,
    filter_chain: str,
    has_audio: bool,
    audio_mode: str,
    overlay_input_args: list[str] | None = None,
    video_map: str = "[vout]",
) -> list[str]:
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(video_path),
    ]
    if overlay_input_args:
        args.extend(overlay_input_args)
    args.extend(["-filter_complex", filter_chain, "-map", video_map, "-fps_mode", "passthrough"])
    if has_audio:
        args.extend(["-map", "0:a?"])
    if overlay_input_args:
        args.append("-shortest")
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


def _log_caption_style(
    context: JobRuntimeContext,
    style: ResolvedCaptionStyle,
    font_resolution: CaptionFontResolution,
) -> None:
    LOGGER.info(
        "%s CAPTION_STYLE renderer=%s font=%s size=%s box=%s shadow=%s radius=%s max_width=%s%%",
        job_prefix(context),
        style.renderer,
        style.font_name,
        style.scaled_font_size,
        "on" if style.box_enabled else "off",
        "on" if style.shadow_enabled else "off",
        style.border_radius,
        style.max_width_percent,
    )
    LOGGER.debug(
        "%s CAPTION_STYLE font_file=%s fallback=%s font_size=%s scale=%.3f colors text=%s bg=%s outline=%s shadow=%s opacity(text/bg/shadow)=%.2f/%.2f/%.2f padding=(%s,%s) margin_v=%s",
        job_prefix(context),
        font_resolution.font_path.name if font_resolution.font_path else "fallback",
        font_resolution.font_name,
        style.font_size,
        style.scale_factor,
        style.text_color,
        style.background_color,
        style.outline_color,
        style.shadow_color,
        style.text_color[3] / 255.0,
        style.background_color[3] / 255.0,
        style.shadow_color[3] / 255.0,
        style.padding_x,
        style.padding_y,
        style.margin_v,
    )
    if font_resolution.warning:
        LOGGER.warning("%s [CAPTION_STYLE] %s", job_prefix(context), font_resolution.warning)


def _caption_scale_factor(width: int, height: int) -> float:
    raw_scale = min(width / 1080.0, height / 1920.0)
    return max(0.6, min(1.5, raw_scale))


def _rgba_from_hex(color: str, opacity: float) -> tuple[int, int, int, int]:
    r, g, b = _parse_hex_colour(color)
    alpha = max(0, min(255, int(round(max(0.0, min(1.0, float(opacity))) * 255))))
    return r, g, b, alpha


def _parse_hex_colour(value: str) -> tuple[int, int, int]:
    raw = value.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return 255, 255, 255
    try:
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    except ValueError:
        return 255, 255, 255


def _import_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFilter, ImageFont
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise RuntimeError(
            "Thiếu Pillow để render rounded caption box. Hãy cài dependency Pillow."
        ) from exc
    return Image, ImageDraw, ImageFilter, ImageFont


def _create_measure_image():
    image_module, _, _, _ = _import_pillow()
    return image_module.new("RGBA", (16, 16), (0, 0, 0, 0))


def _image_draw(image: Any):
    _, image_draw, _, _ = _import_pillow()
    return image_draw.Draw(image)


def _load_font(font_module: Any, style: ResolvedCaptionStyle, size: int, *, warn_on_fallback: bool = True) -> tuple[Any, bool]:
    if style.font_file and style.font_file.exists():
        try:
            return font_module.truetype(str(style.font_file), size), False
        except Exception as exc:
            if warn_on_fallback:
                LOGGER.warning("Không tải được font %s: %s", style.font_file, exc)
    for candidate in (style.font_fallback, "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return font_module.truetype(candidate, size), True
        except Exception:
            continue
    return font_module.load_default(), True


def _fallback_job_context(video_path: Path, output_path: Path) -> JobRuntimeContext:
    return JobRuntimeContext(
        job_id=f"job_{abs(hash(video_path.as_posix())):x}"[:12],
        source="local_input",
        input_path=video_path,
        output_path=output_path,
        worker_slot=None,
        worker_total=None,
        thread_name="main",
        pid=0,
    )
