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


def test_polish_groq_words_generate_full_ass_dialogues(tmp_path: Path) -> None:
    words = [
        _word("Jeśli", 0.02, 0.26),
        _word("ktoś", 0.26, 0.56),
        _word("trzaśnie", 0.56, 0.98),
        _word("drzwiami", 0.98, 1.30),
        _word("swojego", 1.30, 1.68),
        _word("samochodu", 1.68, 2.14),
        _word("o", 2.14, 2.36),
        _word("twój,", 2.36, 2.70),
        _word("istnieje", 2.70, 3.02),
        _word("nieetyczny", 3.02, 3.52),
        _word("sposób,", 3.52, 3.96),
        _word("którego", 3.96, 4.12),
        _word("można", 4.12, 4.44),
        _word("użyć,", 4.44, 4.86),
        _word("aby", 4.86, 4.94),
        _word("się", 4.94, 5.12),
        _word("zemścić.", 5.12, 5.60),
        _word("Najpierw", 5.58, 6.02),
        _word("zrób", 6.02, 6.26),
        _word("zdjęcie", 6.26, 6.64),
        _word("jego", 6.64, 6.84),
        _word("tablicy", 6.84, 7.26),
        _word("rejestracyjnej.", 7.26, 7.92),
        _word("Następnie", 7.92, 8.42),
        _word("wejdź", 8.42, 8.70),
        _word("do", 8.70, 8.82),
        _word("internetu", 8.82, 9.34),
        _word("i", 9.34, 9.48),
        _word("zamów", 9.48, 9.68),
        _word("pełnowymiarową", 9.68, 10.50),
        _word("replikę", 10.50, 10.90),
        _word("dokładnie", 10.90, 11.38),
        _word("tej", 11.38, 11.58),
        _word("samej", 11.58, 11.84),
        _word("tablicy.", 11.84, 12.28),
        _word("Gdy", 12.34, 12.52),
        _word("dotrze", 12.52, 12.74),
        _word("do", 12.74, 12.92),
        _word("twojej", 12.92, 13.20),
        _word("skrzynki", 13.20, 13.50),
        _word("pocztowej,", 13.50, 14.04),
        _word("weź", 14.04, 14.22),
        _word("fałszywą", 14.22, 14.66),
        _word("tablicę.", 14.66, 15.12),
        _word("Podejdź", 15.16, 15.54),
        _word("do", 15.54, 15.66),
        _word("swojego", 15.66, 15.96),
        _word("auta", 15.96, 16.22),
        _word("i", 16.22, 16.40),
        _word("przyklej", 16.40, 16.74),
        _word("ją", 16.74, 16.90),
        _word("na", 16.90, 17.06),
        _word("swoją", 17.06, 17.24),
        _word("prawdziwą", 17.24, 17.74),
        _word("tablicę", 17.74, 18.10),
        _word("rejestracyjną.", 18.10, 18.76),
        _word("Załóż", 18.82, 19.14),
        _word("czapkę,", 19.14, 19.54),
        _word("żeby", 19.54, 19.64),
        _word("kamery", 19.64, 19.98),
        _word("nie", 19.98, 20.10),
        _word("mogły", 20.10, 20.28),
        _word("zobaczyć", 20.28, 20.76),
        _word("twojej", 20.76, 21.08),
        _word("twarzy.", 21.08, 21.36),
        _word("Potem", 21.36, 21.62),
        _word("znajdź", 21.62, 21.96),
        _word("fotoradar", 21.96, 22.44),
        _word("i", 22.44, 22.56),
        _word("przejedź", 22.56, 22.88),
        _word("obok", 22.88, 23.14),
        _word("niego", 23.14, 23.26),
        _word("tak", 23.26, 23.52),
        _word("szybko,", 23.52, 23.90),
        _word("jak", 23.90, 23.98),
        _word("to", 23.98, 24.10),
        _word("możliwe.", 24.10, 24.58),
        _word("Powtórz", 24.52, 24.80),
        _word("to", 24.80, 24.96),
        _word("kilka", 24.96, 25.18),
        _word("razy,", 25.18, 25.58),
        _word("aż", 25.58, 25.68),
        _word("uzbiera", 25.68, 25.98),
        _word("się", 25.98, 26.14),
        _word("kilka", 26.14, 26.36),
        _word("tysięcy", 26.36, 26.74),
        _word("złotych", 26.74, 27.20),
        _word("kar,", 27.20, 27.50),
        _word("a", 27.50, 27.58),
        _word("każdy", 27.58, 27.78),
        _word("mandat", 27.78, 28.14),
        _word("zostanie", 28.14, 28.50),
        _word("wysłany", 28.50, 28.84),
        _word("do", 28.84, 28.98),
        _word("faceta,", 28.98, 29.42),
        _word("który", 29.42, 29.54),
        _word("wgniótł", 29.54, 29.96),
        _word("twoje", 29.96, 30.22),
        _word("auto.", 30.22, 30.58),
    ]
    formatting = FormattingConfig()
    cues = subtitles_module.split_words_into_caption_cues(
        words,
        max_chars_per_line=formatting.max_chars_per_line,
        max_lines=formatting.max_lines,
        max_chars_per_cue=formatting.max_chars_per_cue,
        max_words_per_cue=formatting.max_words_per_cue,
        min_duration=formatting.min_duration,
        max_duration=formatting.max_duration,
        pause_threshold=formatting.pause_threshold,
    )
    cues = subtitles_module.stabilize_caption_cues(
        cues,
        min_duration=formatting.min_duration,
        media_duration=30.58,
    )

    assert len(cues) == 23
    assert cues[0].text == "Jeśli ktoś trzaśnie\ndrzwiami swojego"
    assert cues[-1].text == "faceta, który\nwgniótł twoje auto."

    ass_path = subtitles_module.write_ass(
        cues,
        tmp_path / "polish.ass",
        subtitles_module.AssStyleConfig(
            font_name="PlaywriteGBJ",
            font_size=45,
            outline=0,
            shadow=1,
            margin_v=140,
            alignment="bottom",
            play_res_x=1080,
            play_res_y=1920,
            text_color="#111111",
            outline_color="#000000",
            background_color="#FFFFFF",
            text_opacity=1.0,
            outline_opacity=0.0,
            background_opacity=0.95,
            box_enabled=True,
        ),
    )
    ass_text = ass_path.read_text(encoding="utf-8")

    assert ass_text.count("Dialogue:") == 23
    assert "Jeśli ktoś trzaśnie\\Ndrzwiami swojego" in ass_text
    assert "rejestracyjną. Załóż\\Nczapkę," in ass_text
    assert "faceta, który\\Nwgniótł twoje auto." in ass_text


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


def test_language_override_is_passed_to_transcription(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    (tmp_path / "output" / "subtitles").mkdir(parents=True)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        subtitles_module,
        "probe_video",
        lambda _: VideoInfo(path=Path("x.mp4"), duration=2.0, width=1080, height=1920, fps=30.0, has_audio=True),
    )

    class FakeManager:
        def __init__(self, config, job_context):  # noqa: ANN001
            pass

        def resolve_backend(self) -> str:
            return "groq"

        def transcribe(self, media_path: Path, language: str | None):  # noqa: ANN001
            captured["language"] = language
            return type(
                "Result",
                (),
                {
                    "backend": "groq",
                    "language": "vi",
                    "segments": [],
                    "words": [
                        type("Word", (), {"text": "Xin", "start": 0.0, "end": 0.5})(),
                        type("Word", (), {"text": "chao", "start": 0.55, "end": 1.0})(),
                    ],
                },
            )()

    monkeypatch.setattr(subtitles_module, "TranscriptionManager", FakeManager)

    result = subtitles_module.generate_subtitles_for_video(
        tmp_path / "video.mp4",
        SubtitlesConfig(enabled=True, language="auto"),
        project_root,
        temp_root,
        language_override="vi",
        debug=False,
    )

    assert captured["language"] == "vi"
    assert result is not None
    assert result.language == "vi"


def test_wrap_caption_text_limits_to_two_lines() -> None:
    lines = subtitles_module.wrap_caption_text(
        "Những anh em đi trước hoặc trong ngành cho mình xin lời khuyên",
        max_chars_per_line=20,
        max_lines=2,
    )

    assert len(lines) <= 2
