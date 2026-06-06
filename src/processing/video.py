from __future__ import annotations

import random
from dataclasses import dataclass


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
    duration: float,
    *,
    min_seconds: float = 3.0,
    max_seconds: float = 5.0,
    seed: int | None = None,
) -> list[Segment]:
    # V1 giữ kiến trúc tách riêng cho phát hiện cảnh; hiện tại dùng lại chia đoạn ngẫu nhiên.
    return generate_random_segments(
        duration,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        seed=seed,
    )
