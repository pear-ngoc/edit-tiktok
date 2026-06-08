from pathlib import Path
from unittest.mock import patch

from config import default_config
import app
from models import VideoInfo


def test_wizard_empty_input_stops_before_questions(tmp_path: Path, capsys) -> None:
    config = default_config()

    with (
        patch("app._load_effective_config", return_value=config),
        patch("app.ensure_input_videos_exist", return_value=[]),
        patch("builtins.input", side_effect=AssertionError("input() should not be called")),
        patch("app.run_batch") as mocked_run_batch,
    ):
        app.wizard(tmp_path)

    mocked_run_batch.assert_not_called()
    out = capsys.readouterr().out
    assert "No input videos found. Please add videos to the input/ folder and run again." in out
    assert "Burn captions into the video?" not in out


def test_wizard_unsupported_files_only_stops_before_questions(tmp_path: Path, capsys) -> None:
    config = default_config()

    with (
        patch("app._load_effective_config", return_value=config),
        patch("app.ensure_input_videos_exist", return_value=[]),
        patch("builtins.input", side_effect=AssertionError("input() should not be called")),
        patch("app.run_batch") as mocked_run_batch,
    ):
        app.wizard(tmp_path)

    mocked_run_batch.assert_not_called()
    out = capsys.readouterr().out
    assert "No input videos found. Please add videos to the input/ folder and run again." in out


def test_wizard_with_one_mp4_continues(tmp_path: Path) -> None:
    config = default_config()
    video = tmp_path / "input" / "clip.mp4"
    layout_sample = VideoInfo(
        path=video,
        duration=10.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        display_aspect_ratio="16:9",
    )

    with (
        patch("app._load_effective_config", return_value=config),
        patch("app.ensure_input_videos_exist", return_value=[video]),
        patch("app._probe_layout_sample_info", return_value=layout_sample),
        patch("app._configure_subtitles_interactively"),
        patch("app._configure_luts_interactively"),
        patch("builtins.input", side_effect=[""]),
        patch("app._ask", side_effect=lambda prompt, default: default),
        patch("app._ask_bool", side_effect=lambda prompt, default: default),
        patch("app._print_wizard_summary"),
        patch("app._maybe_save_wizard_profile"),
        patch("app.ensure_runtime_dirs"),
        patch("app.configure_logging"),
        patch("app.run_batch") as mocked_run_batch,
    ):
        app.wizard(tmp_path)

    mocked_run_batch.assert_called_once()
    assert config.video.center_crop_blur.foreground_aspect_ratio == "4:3"


def test_wizard_recursive_and_non_recursive_input_scan(tmp_path: Path) -> None:
    config = default_config()
    input_root = tmp_path / "input"
    nested = input_root / "nested"
    nested.mkdir(parents=True)
    video = nested / "clip.webm"
    video.write_bytes(b"fake")

    config.processing.recursive = True
    with patch("config.ensure_config_file", return_value=tmp_path / "config.yaml"):
        recursive_videos = app.ensure_input_videos_exist(tmp_path, config)

    assert recursive_videos == [video]

    config.processing.recursive = False
    with patch("config.ensure_config_file", return_value=tmp_path / "config.yaml"):
        non_recursive_videos = app.ensure_input_videos_exist(tmp_path, config)

    assert non_recursive_videos == []
