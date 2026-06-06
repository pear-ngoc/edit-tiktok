from pathlib import Path

from config import default_config
from models import AppConfig
from processing.preflight import discover_input_videos, print_preflight_summary, run_preflight_checks


def _config() -> AppConfig:
    return default_config()


def test_empty_input_folder_stops_cleanly(tmp_path: Path, capsys) -> None:
    config = _config()

    result = run_preflight_checks(tmp_path, config)

    assert result.total_count == 0
    assert result.videos == []

    print_preflight_summary(result)
    out = capsys.readouterr().out
    assert "No input videos found. Please add videos to the input/ folder and run again." in out


def test_unsupported_files_only_are_ignored(tmp_path: Path) -> None:
    config = _config()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "notes.txt").write_text("hello", encoding="utf-8")
    (input_dir / "image.png").write_text("fake image", encoding="utf-8")

    videos = discover_input_videos(tmp_path, config)

    assert videos == []


def test_one_mp4_continues(tmp_path: Path) -> None:
    config = _config()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    video = input_dir / "clip.mp4"
    video.write_bytes(b"fake")

    videos = discover_input_videos(tmp_path, config)

    assert videos == [video]


def test_recursive_mode_finds_videos_in_subfolders(tmp_path: Path) -> None:
    config = _config()
    config.processing.recursive = True
    nested_dir = tmp_path / "input" / "nested"
    nested_dir.mkdir(parents=True)
    video = nested_dir / "clip.mov"
    video.write_bytes(b"fake")

    videos = discover_input_videos(tmp_path, config)

    assert videos == [video]


def test_non_recursive_mode_ignores_videos_in_subfolders(tmp_path: Path) -> None:
    config = _config()
    config.processing.recursive = False
    nested_dir = tmp_path / "input" / "nested"
    nested_dir.mkdir(parents=True)
    video = nested_dir / "clip.mkv"
    video.write_bytes(b"fake")

    videos = discover_input_videos(tmp_path, config)

    assert videos == []
