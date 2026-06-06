from __future__ import annotations

import json
from pathlib import Path

from ffmpeg_tools.runner import run_command
from models import VideoInfo


def probe_video(path: Path, ffprobe_bin: str = "ffprobe") -> VideoInfo:
    result = run_command(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)

    duration = _float_or_zero(video_stream.get("duration")) or _float_or_zero(
        data.get("format", {}).get("duration")
    )
    return VideoInfo(
        path=path,
        duration=duration,
        width=int(video_stream.get("width") or 0),
        height=int(video_stream.get("height") or 0),
        fps=_parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        has_audio=audio_stream is not None,
        video_codec=str(video_stream.get("codec_name") or ""),
        audio_codec=str(audio_stream.get("codec_name") or "") if audio_stream else "",
        sample_aspect_ratio=str(video_stream.get("sample_aspect_ratio") or ""),
        display_aspect_ratio=str(video_stream.get("display_aspect_ratio") or ""),
        time_base=str(video_stream.get("time_base") or ""),
        audio_sample_rate=int(audio_stream.get("sample_rate") or 0) if audio_stream else 0,
        audio_channels=int(audio_stream.get("channels") or 0) if audio_stream else 0,
        audio_bitrate=int(audio_stream.get("bit_rate") or 0) if audio_stream else 0,
        audio_duration=_float_or_zero(audio_stream.get("duration")) if audio_stream else 0.0,
        video_bitrate=int(video_stream.get("bit_rate") or 0),
    )


def _parse_fps(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denom = float(denominator)
        return float(numerator) / denom if denom else 0.0
    return float(value)


def _float_or_zero(value: object) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
