from __future__ import annotations

from pathlib import Path

import pytest

from config import default_config
from models import AppConfig, EncoderSelection, FormattingConfig, SubtitleCue, VideoInfo
from processing.caption_renderer import (
    build_caption_overlay_filter,
    available_caption_fonts,
    burn_rounded_captions,
    render_caption_image,
    resolve_caption_font,
    resolve_caption_style,
)


def test_custom_ttf_font_resolves_from_assets_font(tmp_path: Path) -> None:
    font_dir = tmp_path / "assets" / "font"
    font_dir.mkdir(parents=True)
    font_file = font_dir / "BeVietnamPro-Bold.ttf"
    font_file.write_bytes(b"font")

    formatting = FormattingConfig(caption_font_file="BeVietnamPro-Bold.ttf")
    resolved = resolve_caption_font(tmp_path, formatting)

    assert resolved.font_path == font_file.resolve()


def test_missing_font_uses_fallback(tmp_path: Path) -> None:
    formatting = FormattingConfig(caption_font_file="Missing.ttf", caption_font_fallback="Arial")
    resolved = resolve_caption_font(tmp_path, formatting)

    assert resolved.font_path is None
    assert resolved.font_name == "Arial"
    assert resolved.warning is not None


def test_unsafe_font_paths_are_rejected(tmp_path: Path) -> None:
    font_dir = tmp_path / "assets" / "font"
    font_dir.mkdir(parents=True)
    formatting = FormattingConfig(caption_font_file="../../evil.ttf", caption_font_fallback="Arial")

    resolved = resolve_caption_font(tmp_path, formatting)

    assert resolved.font_path is None
    assert "assets/font" in (resolved.warning or "")


def test_opacity_semantics_are_not_inverted(tmp_path: Path) -> None:
    style, _ = resolve_caption_style(
        FormattingConfig(
            caption_renderer="rounded_box",
            caption_text_color="#111111",
            caption_text_opacity=1.0,
            caption_background_color="#FFFFFF",
            caption_background_opacity=0.95,
            caption_shadow_color="#000000",
            caption_shadow_opacity=0.25,
        ),
        video_width=1080,
        video_height=1920,
        project_root=tmp_path,
    )

    assert style.text_color[-1] == 255
    assert style.background_color[-1] == 242
    assert style.shadow_color[-1] == 64


def test_available_caption_fonts_lists_only_ttf_otf(tmp_path: Path) -> None:
    font_dir = tmp_path / "assets" / "font"
    font_dir.mkdir(parents=True)
    (font_dir / "A.ttf").write_text("", encoding="utf-8")
    (font_dir / "B.otf").write_text("", encoding="utf-8")
    (font_dir / "C.txt").write_text("", encoding="utf-8")

    fonts = available_caption_fonts(tmp_path)

    assert [path.name for path in fonts] == ["A.ttf", "B.otf"]


def test_overlay_filter_uses_one_input_per_caption() -> None:
    assets = [
        type("Asset", (), {"image_path": Path("cue_0001.png"), "start": 0.0, "end": 1.2})(),
        type("Asset", (), {"image_path": Path("cue_0002.png"), "start": 1.3, "end": 2.1})(),
    ]

    input_args, filter_complex, final_label = build_caption_overlay_filter(
        assets,
        margin_v=120,
        vertical_offset=0,
    )

    assert input_args == ["-loop", "1", "-i", "cue_0001.png", "-loop", "1", "-i", "cue_0002.png"]
    assert "overlay=" in filter_complex
    assert "setpts" not in filter_complex
    assert final_label == "[v2]"


def test_render_caption_image_creates_transparent_border_and_white_center(tmp_path: Path) -> None:
    PIL = pytest.importorskip("PIL")  # noqa: N806
    from PIL import Image

    _ = PIL

    cue = SubtitleCue(
        start=0.0,
        end=1.5,
        text="Những anh em đi trước\nhoặc trong ngành",
        lines=["Những anh em đi trước", "hoặc trong ngành"],
    )
    style, _ = resolve_caption_style(
        default_config().formatting,
        video_width=1080,
        video_height=1920,
        project_root=tmp_path,
    )
    image_path = tmp_path / "caption.png"

    render_caption_image(cue, style, image_path)

    assert image_path.exists()

    img = Image.open(image_path).convert("RGBA")
    px = img.load()

    box_points: list[tuple[int, int]] = []
    text_points: list[tuple[int, int]] = []
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = px[x, y]
            if a > 220:
                if r > 240 and g > 240 and b > 240:
                    box_points.append((x, y))
                elif r < 120 and g < 120 and b < 120:
                    text_points.append((x, y))

    def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return min(xs), min(ys), max(xs) + 1, max(ys) + 1

    box_bbox = _bbox(box_points)
    text_bbox = _bbox(text_points)

    assert abs(text_bbox[0] - (box_bbox[0] + style.padding_x)) <= 2
    assert abs(text_bbox[1] - (box_bbox[1] + style.padding_y)) <= 2


def test_burn_rounded_captions_builds_overlay_command_and_cleans_temp(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "input.mp4"
    video.write_text("video", encoding="utf-8")
    output = tmp_path / "output.mp4"
    project_root = tmp_path
    temp_root = tmp_path / "temp"
    cues = [
        SubtitleCue(start=0.0, end=1.0, text="Xin chào", lines=["Xin chào"]),
        SubtitleCue(start=1.2, end=2.2, text="mọi người", lines=["mọi người"]),
    ]

    monkeypatch.setattr(
        "processing.caption_renderer.probe_video",
        lambda path: VideoInfo(
            path=path,
            duration=2.2,
            width=1080,
            height=1920,
            fps=30.0,
            has_audio=True,
            video_codec="h264",
            audio_codec="aac",
            sample_aspect_ratio="1:1",
            display_aspect_ratio="9:16",
            time_base="1/15360",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=192000,
            audio_duration=2.2,
            video_bitrate=4000000,
        ),
    )
    monkeypatch.setattr("processing.caption_renderer.detect_available_encoders", lambda: ["libx264"])
    monkeypatch.setattr(
        "processing.caption_renderer.select_encoder",
        lambda *args, **kwargs: EncoderSelection(
            backend="cpu_h264",
            codec_name="libx264",
            args=["-c:v", "libx264", "-preset", "medium", "-crf", "20"],
            description="cpu_h264 dùng libx264",
        ),
    )
    rendered_files: list[Path] = []

    def fake_render_caption_image(cue, style, output_path):  # noqa: ANN001
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{cue.text}", encoding="utf-8")
        rendered_files.append(output_path)
        return output_path

    monkeypatch.setattr("processing.caption_renderer.render_caption_image", fake_render_caption_image)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "processing.caption_renderer.run_command",
        lambda args, debug=False, stderr_tail_lines=40: commands.append(list(args)),
    )
    monkeypatch.setattr(
        "processing.caption_renderer.probe_video",
        lambda path: VideoInfo(
            path=path,
            duration=2.2,
            width=1080,
            height=1920,
            fps=30.0,
            has_audio=True,
            video_codec="h264",
            audio_codec="aac",
            sample_aspect_ratio="1:1",
            display_aspect_ratio="9:16",
            time_base="1/15360",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=192000,
            audio_duration=2.2,
            video_bitrate=4000000,
        ),
    )

    result = burn_rounded_captions(
        video,
        output,
        cues,
        AppConfig(formatting=FormattingConfig(caption_renderer="rounded_box")),
        project_root=project_root,
        temp_root=temp_root,
        debug=False,
    )

    assert result == output
    assert commands
    joined = " ".join(commands[0])
    assert "overlay=" in joined
    assert "-shortest" in commands[0]
    assert "setpts" not in joined
    assert "atempo" not in joined
    assert "asetrate" not in joined
    assert "-c:a" in commands[0]
    assert "copy" in commands[0]
    assert not any(path.exists() for path in rendered_files)
