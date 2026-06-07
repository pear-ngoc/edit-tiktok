from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from processing.video import generate_scene_segments


def test_generate_scene_segments_uses_scene_times(monkeypatch) -> None:
    def fake_run_command(args, *, check=False):  # noqa: ANN001
        return SimpleNamespace(
            returncode=0,
            stderr="\n".join(
                [
                    "[Parsed_showinfo_0 @ 0x] n:0 pts:0 pts_time:0.000000",
                    "[Parsed_showinfo_0 @ 0x] n:1 pts:15360 pts_time:1.000000",
                    "[Parsed_showinfo_0 @ 0x] n:2 pts:30720 pts_time:2.000000",
                    "[Parsed_showinfo_0 @ 0x] n:3 pts:46080 pts_time:3.000000",
                ]
            ),
        )

    monkeypatch.setattr("processing.video.run_command", fake_run_command)
    segments = generate_scene_segments(Path("video.mp4"), 4.0, scene_threshold=0.3)
    assert [(round(item.start, 3), round(item.end, 3), item.index) for item in segments] == [
        (0.0, 1.0, 0),
        (1.0, 2.0, 1),
        (2.0, 3.0, 2),
        (3.0, 4.0, 3),
    ]
