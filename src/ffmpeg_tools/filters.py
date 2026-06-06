from __future__ import annotations

from pathlib import Path


def parse_aspect_ratio(value: str) -> tuple[int, int]:
    normalized = value.lower().strip().replace("x", ":")
    if ":" not in normalized:
        raise ValueError(f"Tỷ lệ khung hình không hợp lệ: {value}")
    left, right = normalized.split(":", 1)
    width_ratio = int(left)
    height_ratio = int(right)
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError(f"Tỷ lệ khung hình không hợp lệ: {value}")
    return width_ratio, height_ratio


def parse_target_resolution(value: str, aspect_ratio: str) -> tuple[int, int]:
    aspect_w, aspect_h = parse_aspect_ratio(aspect_ratio)
    normalized = value.lower().strip()
    if normalized in {"original", "source", "keep"}:
        return 0, 0
    if "x" in normalized:
        width, height = normalized.split("x", 1)
        return _even(int(width)), _even(int(height))
    if normalized.endswith("p"):
        long_side = int(normalized[:-1])
        if aspect_h >= aspect_w:
            height = long_side
            width = round(height * aspect_w / aspect_h)
        else:
            width = long_side
            height = round(width * aspect_h / aspect_w)
        return _even(width), _even(height)
    raise ValueError(f"Độ phân giải mục tiêu không hợp lệ: {value}")


def choose_output_resolution(
    source_width: int,
    source_height: int,
    aspect_ratio: str,
    target_resolution: str,
    keep_original: bool,
) -> tuple[int, int]:
    if keep_original:
        return _even(source_width), _even(source_height)
    width, height = parse_target_resolution(target_resolution, aspect_ratio)
    if width and height:
        return width, height
    return _even(source_width), _even(source_height)


def build_crop_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1"
    )


def build_cinematic_blur_filter(width: int, height: int, blur_sigma: int = 30) -> str:
    return (
        f"split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma={blur_sigma}[bg];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def build_keep_or_target_filter(width: int, height: int) -> str:
    return f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"


def build_segment_video_filter(
    *,
    mode: str,
    width: int,
    height: int,
    zoom: float,
    horizontal_flip: bool,
    fade_seconds: float,
    duration: float,
    contrast: float,
    saturation: float,
    sharpen: bool,
    noise_overlay: bool,
    noise_alpha: float,
) -> str:
    filters: list[str] = [build_base_video_filter(mode, width, height)]
    segment_transform = build_segment_transform_filter(
        zoom=zoom,
        horizontal_flip=horizontal_flip,
        fade_seconds=fade_seconds,
        duration=duration,
    )
    if segment_transform:
        filters.append(segment_transform)
    filters.append(build_color_adjust_filter(contrast, saturation, sharpen))
    if noise_overlay:
        filters.append(build_noise_overlay_filter(noise_alpha))
    filters.append("format=yuv420p")
    return ",".join(filters)


def build_segment_transform_filter(
    *,
    zoom: float = 1.0,
    horizontal_flip: bool = False,
    fade_seconds: float = 0.0,
    duration: float = 0.0,
) -> str:
    filters: list[str] = []
    if zoom and zoom != 1.0:
        filters.append(f"scale=iw*{zoom}:ih*{zoom},crop=iw/{zoom}:ih/{zoom}")
    if horizontal_flip:
        filters.append("hflip")
    if fade_seconds > 0:
        filters.append(f"fade=t=in:st=0:d={fade_seconds}")
        if duration > fade_seconds:
            filters.append(f"fade=t=out:st={max(0, duration - fade_seconds)}:d={fade_seconds}")
    return ",".join(filters)


def build_lut_filter(lut_paths: list[Path]) -> str:
    return ",".join(f"lut3d=file='{escape_filter_path(path)}'" for path in lut_paths)


def build_color_adjust_filter(contrast: float, saturation: float, sharpen: bool) -> str:
    filters = [f"eq=contrast={contrast}:saturation={saturation}"]
    if sharpen:
        filters.append("unsharp=5:5:0.8:3:3:0.4")
    return ",".join(filters)


def build_audio_filter(
    *,
    volume: float,
    speed: float,
    tempo_match_speed: bool,
    pitch_shift_semitones: float,
    random_eq: bool = False,
    bass_gain: float = 0,
    treble_gain: float = 0,
) -> str:
    filters: list[str] = [f"volume={volume}"]
    if tempo_match_speed and speed != 1.0:
        filters.extend(_atempo_chain(speed))
    if pitch_shift_semitones:
        factor = 2 ** (pitch_shift_semitones / 12)
        filters.append(f"asetrate=48000*{factor}")
        filters.append("aresample=48000")
        filters.extend(_atempo_chain(1 / factor))
    if random_eq:
        filters.append(f"bass=g={bass_gain}")
        filters.append(f"treble=g={treble_gain}")
    return ",".join(filters)


def build_ambient_mix_filter(input_count: int, duration: float) -> str:
    return f"amix=inputs={input_count}:duration=longest:dropout_transition=2,atrim=0:{duration},asetpts=N/SR/TB"


def build_speed_filter(speed: float) -> str:
    if speed <= 0:
        raise ValueError("Tốc độ phải lớn hơn 0")
    return f"setpts=PTS/{speed}"


def build_noise_overlay_filter(noise_alpha: float) -> str:
    strength = max(1, min(12, round(noise_alpha * 400)))
    return f"noise=alls={strength}:allf=t"


def build_subtitles_burn_filter(
    subtitle_path: Path,
    *,
    font_size: int = 24,
    margin_v: int = 48,
) -> str:
    escaped_path = escape_filter_path(subtitle_path)
    force_style = (
        "Alignment=2,"
        f"MarginV={margin_v},"
        "Outline=2,"
        "Shadow=0,"
        f"Fontsize={font_size},"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000"
    )
    return f"subtitles=filename='{escaped_path}':force_style='{force_style}'"


def build_ass_burn_filter(subtitle_path: Path) -> str:
    escaped_path = escape_filter_path(subtitle_path)
    return f"ass=filename='{escaped_path}'"


def escape_filter_path(path: Path) -> str:
    return path.as_posix().replace("\\", "\\\\").replace(":", "\\:").replace(",", "\\,").replace("'", "\\'")


def _atempo_chain(speed: float) -> list[str]:
    if speed <= 0:
        raise ValueError("Tốc độ âm thanh phải lớn hơn 0")
    values: list[float] = []
    remaining = speed
    while remaining > 2.0:
        values.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        values.append(0.5)
        remaining /= 0.5
    values.append(remaining)
    return [f"atempo={value:.6g}" for value in values]


def _even(value: int) -> int:
    return max(2, value - (value % 2))


def build_base_video_filter(mode: str, width: int, height: int) -> str:
    normalized = mode.lower()
    if normalized in {"crop", "simple_crop"}:
        return build_crop_filter(width, height)
    if normalized in {"blur", "cinematic_blur"}:
        return build_cinematic_blur_filter(width, height)
    if normalized in {"original", "keep", "target"}:
        return build_keep_or_target_filter(width, height)
    raise ValueError(f"Chế độ video không được hỗ trợ: {mode}")
