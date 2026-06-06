from pathlib import Path

from ffmpeg_tools.filters import (
    build_color_adjust_filter,
    build_crop_filter,
    build_lut_filter,
    parse_aspect_ratio,
    parse_target_resolution,
)


def test_parse_aspect_ratio() -> None:
    assert parse_aspect_ratio("9:16") == (9, 16)
    assert parse_aspect_ratio("16x9") == (16, 9)


def test_parse_target_resolution_vertical() -> None:
    assert parse_target_resolution("720p", "9:16") == (404, 720)


def test_build_filter_strings() -> None:
    assert "crop=1080:1920" in build_crop_filter(1080, 1920)
    assert "eq=contrast=1.03:saturation=1.08" in build_color_adjust_filter(1.03, 1.08, True)
    chain = build_lut_filter([Path("assets/luts/look.cube"), Path("assets/luts/look2.cube")])
    assert "lut3d=file=" in chain
    assert chain.count("lut3d=file=") == 2
