from pathlib import Path

import pytest

from ffmpeg_tools.encoders import EncoderSelection
from ffmpeg_tools.filters import build_ass_burn_filter, build_subtitles_burn_filter
from ffmpeg_tools.runner import CommandResult
from models import AppConfig, FormattingConfig, SubtitlesConfig, VideoInfo
from processing import subtitles as subtitles_module


def _word(text: str, start: float, end: float) -> subtitles_module.SubtitleWord:
    return subtitles_module.SubtitleWord(text=text, start=start, end=end)


def test_srt_timestamp_formatting() -> None:
    assert subtitles_module.format_srt_timestamp(0) == "00:00:00,000"
    assert subtitles_module.format_srt_timestamp(61.234) == "00:01:01,234"


def test_split_long_sentence_into_multiple_cues() -> None:
    words = [
        _word("Những", 0.0, 0.18),
        _word("anh", 0.20, 0.32),
        _word("em", 0.34, 0.45),
        _word("đi", 0.48, 0.56),
        _word("trước", 0.58, 0.72),
        _word("hoặc", 0.75, 0.90),
        _word("trong", 0.93, 1.05),
        _word("ngành", 1.08, 1.24),
        _word("cho", 1.28, 1.40),
        _word("mình", 1.43, 1.58),
        _word("xin", 1.61, 1.75),
        _word("lời", 1.78, 1.90),
        _word("khuyên", 1.93, 2.10),
        _word("để", 2.14, 2.22),
        _word("mình", 2.25, 2.38),
        _word("thay", 2.41, 2.55),
        _word("đổi", 2.58, 2.72),
        _word("được", 2.75, 2.92),
    ]
    cues = subtitles_module.split_words_into_caption_cues(
        words,
        max_chars_per_line=20,
        max_lines=2,
        max_chars_per_cue=40,
        max_words_per_cue=7,
        min_duration=0.7,
        max_duration=2.6,
        pause_threshold=0.45,
    )

    assert len(cues) >= 2
    assert all(len(cue.lines) <= 2 for cue in cues)
    assert any(len(cue.text.replace("\n", " ")) <= 40 for cue in cues)


def test_punctuation_creates_preferred_break() -> None:
    words = [
        _word("Xin", 0.0, 0.2),
        _word("chào,", 0.22, 0.45),
        _word("bạn", 0.50, 0.65),
    ]
    cues = subtitles_module.split_words_into_caption_cues(
        words,
        max_chars_per_line=20,
        max_lines=2,
        max_chars_per_cue=40,
        max_words_per_cue=7,
        min_duration=0.7,
        max_duration=2.6,
        pause_threshold=0.45,
    )

    assert len(cues) == 2
    assert cues[0].text.startswith("Xin chào,")
    assert cues[1].text == "bạn"


def test_pause_creates_break() -> None:
    words = [
        _word("Xin", 0.0, 0.2),
        _word("chào", 0.25, 0.4),
        _word("mọi", 1.0, 1.1),
        _word("người", 1.15, 1.3),
    ]
    cues = subtitles_module.split_words_into_caption_cues(
        words,
        max_chars_per_line=20,
        max_lines=2,
        max_chars_per_cue=40,
        max_words_per_cue=7,
        min_duration=0.7,
        max_duration=2.6,
        pause_threshold=0.45,
    )

    assert len(cues) == 2
    assert cues[0].text == "Xin chào"
    assert cues[1].text == "mọi người"


def test_stabilize_caption_cues_extends_short_single_cue() -> None:
    cues = subtitles_module.stabilize_caption_cues(
        [
            subtitles_module.SubtitleCue(
                start=0.46,
                end=0.70,
                text="You",
                lines=["You"],
            )
        ],
        min_duration=0.7,
        media_duration=21.87,
    )

    assert len(cues) == 1
    assert cues[0].start == 0.46
    assert cues[0].end == pytest.approx(1.16)


def test_build_segment_caption_cues_creates_lines_without_word_timestamps() -> None:
    cues = subtitles_module.build_segment_caption_cues(
        [
            {"text": "Xin chao moi nguoi", "start": 0.0, "end": 1.5},
            {"text": "Chung ta tiep tuc", "start": 1.7, "end": 3.0},
        ],
        max_chars_per_line=20,
        max_lines=2,
    )

    assert len(cues) == 2
    assert cues[0].text
    assert cues[0].end == 1.5


def test_overlay_filter_uses_eof_action_pass() -> None:
    from processing.caption_renderer import RenderedCaptionAsset, build_caption_overlay_filter

    _, filter_complex, _ = build_caption_overlay_filter(
        [RenderedCaptionAsset(image_path=Path("cue_0001.png"), start=0.0, end=1.0)],
        margin_v=140,
        vertical_offset=0,
    )

    assert "eof_action=pass" in filter_complex


def test_write_srt_and_vtt(tmp_path: Path) -> None:
    entries = [
        subtitles_module.SubtitleCue(
            start=0.0,
            end=1.5,
            text="Xin chào\nthế giới",
            lines=["Xin chào", "thế giới"],
        ),
        subtitles_module.SubtitleCue(
            start=1.6,
            end=3.0,
            text="Chúng ta đi tiếp",
            lines=["Chúng ta đi", "tiếp"],
        ),
    ]
    srt = subtitles_module.write_srt(entries, tmp_path / "sample.srt")
    vtt = subtitles_module.write_vtt(entries, tmp_path / "sample.vtt")

    srt_text = srt.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,500" in srt_text
    assert "Xin chào\nthế giới" in srt_text
    assert "WEBVTT" in vtt.read_text(encoding="utf-8")


def test_write_ass_contains_style_and_dialogue(tmp_path: Path) -> None:
    cues = [
        subtitles_module.SubtitleCue(
            start=0.0,
            end=1.0,
            text="Xin chào\nmọi người",
            lines=["Xin chào", "mọi người"],
        )
    ]
    ass_path = subtitles_module.write_ass(
        cues,
        tmp_path / "sample.ass",
        subtitles_module.AssStyleConfig(
            font_name="Be Vietnam Pro",
            font_size=64,
            outline=6,
            shadow=2,
            margin_v=180,
            alignment="bottom",
            play_res_x=404,
            play_res_y=720,
            text_color="#FFEEAA",
            outline_color="#111111",
            background_color="#000000",
            text_opacity=1.0,
            outline_opacity=1.0,
            background_opacity=0.5,
            box_enabled=True,
        ),
    )
    text = ass_path.read_text(encoding="utf-8")
    assert "[V4+ Styles]" in text
    assert "Style: TikTok" in text
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,TikTok" in text
    assert r"\N" in text
    assert "PlayResX: 404" in text
    assert "PlayResY: 720" in text
    assert "Be Vietnam Pro" in text
    assert "&H00AAEEFF" in text
    assert "&H00111111" in text
    assert "&H80000000" in text
    assert "BorderStyle" in text
    assert ",3," in text


def test_style_config_supports_vertical_offset_and_colors() -> None:
    formatting = FormattingConfig(
        caption_position="bottom",
        caption_vertical_offset=30,
        caption_font_name="Roboto",
        caption_font_size=62,
        caption_text_color="#FFFFFF",
        caption_outline_color="#000000",
        caption_background_color="#000000",
        caption_text_opacity=1.0,
        caption_outline_opacity=1.0,
        caption_background_opacity=0.6,
        caption_outline=5,
        caption_shadow=1,
        caption_margin_v=120,
        caption_box_enabled=True,
    )
    style = subtitles_module._style_config_from_formatting(formatting, play_res_x=404, play_res_y=720)

    assert style.font_name == "Roboto"
    assert style.margin_v == 150
    assert style.play_res_x == 404
    assert style.play_res_y == 720
    assert style.box_enabled is True


def test_burn_filter_contains_ass_path(tmp_path: Path) -> None:
    path = tmp_path / "sample.ass"
    path.write_text("", encoding="utf-8")
    filter_chain = build_ass_burn_filter(path)
    assert "ass=" in filter_chain


def test_burn_command_contains_ass_filter_and_preserves_streams(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "input.mp4"
    video.write_text("video", encoding="utf-8")
    subtitle = tmp_path / "sample.ass"
    subtitle.write_text("", encoding="utf-8")
    output = tmp_path / "output.mp4"

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        subtitles_module,
        "probe_video",
        lambda _: VideoInfo(
            path=video,
            duration=1.0,
            width=404,
            height=720,
            fps=30.0,
            has_audio=True,
            video_codec="h264",
            audio_codec="aac",
            sample_aspect_ratio="1:1",
            display_aspect_ratio="404:720",
            time_base="1/15360",
            audio_sample_rate=44100,
            audio_channels=2,
            audio_bitrate=192000,
            audio_duration=1.0,
            video_bitrate=4000000,
        ),
    )
    monkeypatch.setattr(
        subtitles_module,
        "detect_available_encoders",
        lambda: ["libx264"],
    )
    monkeypatch.setattr(
        subtitles_module,
        "select_encoder",
        lambda *args, **kwargs: EncoderSelection(
            backend="cpu_h264",
            codec_name="libx264",
            args=["-c:v", "libx264", "-preset", "medium", "-crf", "20"],
            description="cpu_h264 dùng libx264",
        ),
    )
    monkeypatch.setattr(
        subtitles_module,
        "run_command",
        lambda args, debug=False, check=True, stderr_tail_lines=40: captured.setdefault("args", args),
    )

    result = subtitles_module.burn_subtitles_into_video(
        video,
        subtitle,
        output,
        AppConfig(formatting=FormattingConfig(caption_renderer="ass")),
        debug=False,
    )

    assert result == output
    assert "-vf" in captured["args"]
    assert any("ass=" in part for part in captured["args"])
    assert "scale=" not in " ".join(captured["args"])
    assert "crop=" not in " ".join(captured["args"])
    assert "setpts=" not in " ".join(captured["args"])
    assert "atempo=" not in " ".join(captured["args"])
    assert "asetrate=" not in " ".join(captured["args"])
    assert "-c:a" in captured["args"]
    assert "copy" in captured["args"]
    assert "-map_metadata" in captured["args"]
    assert "-map_chapters" in captured["args"]
    assert "-fps_mode" in captured["args"]
    assert "passthrough" in captured["args"]
    assert "copy" in captured["args"]
    assert "aac" not in captured["args"]


def test_burn_command_falls_back_to_aac_when_copy_fails(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "input.mp4"
    video.write_text("video", encoding="utf-8")
    subtitle = tmp_path / "sample.ass"
    subtitle.write_text("", encoding="utf-8")
    output = tmp_path / "output.mp4"

    calls: list[list[str]] = []

    monkeypatch.setattr(
        subtitles_module,
        "probe_video",
        lambda _: VideoInfo(
            path=video,
            duration=1.0,
            width=404,
            height=720,
            fps=30.0,
            has_audio=True,
        ),
    )
    monkeypatch.setattr(subtitles_module, "detect_available_encoders", lambda: ["libx264"])
    monkeypatch.setattr(
        subtitles_module,
        "select_encoder",
        lambda *args, **kwargs: EncoderSelection(
            backend="cpu_h264",
            codec_name="libx264",
            args=["-c:v", "libx264", "-preset", "medium", "-crf", "20"],
            description="cpu_h264 dùng libx264",
        ),
    )

    def fake_run_command(args, debug=False, check=True, stderr_tail_lines=40):
        calls.append(list(args))
        if len(calls) == 1:
            raise subtitles_module.FFmpegError(
                "copy failed",
                CommandResult(args=list(args), returncode=1, stdout="", stderr=""),
            )
        return CommandResult(args=list(args), returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subtitles_module, "run_command", fake_run_command)

    subtitles_module.burn_subtitles_into_video(
        video,
        subtitle,
        output,
        AppConfig(formatting=FormattingConfig(caption_renderer="ass")),
        debug=False,
    )

    assert len(calls) == 2
    assert "copy" in calls[0]
    assert "aac" in calls[1]
    assert "atempo=" not in " ".join(calls[1])
    assert "asetrate=" not in " ".join(calls[1])


def test_no_audio_skips_subtitle_generation(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    (tmp_path / "output" / "subtitles").mkdir(parents=True)

    monkeypatch.setattr(
        subtitles_module,
        "probe_video",
        lambda _: VideoInfo(path=Path("x.mp4"), duration=1.0, width=1920, height=1080, fps=30.0, has_audio=False),
    )

    result = subtitles_module.generate_subtitles_for_video(
        tmp_path / "video.mp4",
        SubtitlesConfig(enabled=True),
        project_root,
        temp_root,
        debug=False,
    )

    assert result is None


def test_wrap_caption_text_limits_to_two_lines() -> None:
    lines = subtitles_module.wrap_caption_text(
        "Những anh em đi trước hoặc trong ngành cho mình xin lời khuyên",
        max_chars_per_line=20,
        max_lines=2,
    )

    assert len(lines) <= 2
