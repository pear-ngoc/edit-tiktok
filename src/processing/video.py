from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path

from ffmpeg_tools.runner import run_command


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    index: int


def generate_random_segments(
    duration: float,
    *,
    min_seconds: float = 3.0,
    max_seconds: float = 5.0,
    seed: int | None = None,
) -> list[Segment]:
    if duration <= 0:
        return []
    rng = random.Random(seed)
    segments: list[Segment] = []
    cursor = 0.0
    index = 0
    while cursor < duration:
        length = rng.uniform(min_seconds, max_seconds)
        end = min(duration, cursor + length)
        segments.append(Segment(start=cursor, end=end, index=index))
        cursor = end
        index += 1
    return segments


def generate_scene_segments(
    video_path: Path,
    duration: float,
    *,
    scene_threshold: float = 0.3,
    min_scene_gap_seconds: float = 0.5,
    min_seconds: float = 3.0,
    max_seconds: float = 5.0,
    seed: int | None = None,
) -> list[Segment]:
    scene_times = _detect_scene_cut_times(video_path, scene_threshold)
    if not scene_times:
        return generate_random_segments(
            duration,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            seed=seed,
        )

    boundaries = [0.0]
    last_boundary = 0.0
    for scene_time in scene_times:
        if scene_time <= 0.0 or scene_time >= duration:
            continue
        if scene_time - last_boundary < min_scene_gap_seconds:
            continue
        boundaries.append(scene_time)
        last_boundary = scene_time

    if duration > boundaries[-1]:
        boundaries.append(duration)

    segments: list[Segment] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if end <= start:
            continue
        segments.append(Segment(start=start, end=end, index=index))

    if not segments:
        return generate_random_segments(
            duration,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            seed=seed,
        )
    return segments


def _detect_scene_cut_times(video_path: Path, threshold: float) -> list[float]:
    args = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        result = run_command(args, check=False)
    except Exception:
        return []

    times: list[float] = []
    for line in result.stderr.splitlines():
        if "showinfo" not in line or "pts_time:" not in line:
            continue
        match = re.search(r"pts_time:(?P<time>\d+(?:\.\d+)?)", line)
        if not match:
            continue
        try:
            times.append(float(match.group("time")))
        except ValueError:
            continue
    return times
