from pathlib import Path

from config import default_config
from utils.cleanup import clear_workspace
from utils.paths import ensure_runtime_dirs


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_clear_workspace_removes_input_and_generated_dirs(tmp_path: Path) -> None:
    config = default_config()
    ensure_runtime_dirs(tmp_path, config)

    _write(tmp_path / "input" / "a.mp4")
    _write(tmp_path / "output" / "a.mp4")
    _write(tmp_path / "temp" / "tmp.txt")
    _write(tmp_path / "logs" / "video_processing.log")
    _write(tmp_path / "data" / "jobs.json")
    _write(tmp_path / "failed" / "bad.mp4")
    _write(tmp_path / "completed" / "done.mp4")
    _write(tmp_path / "configs" / "profile.yaml")
    _write(tmp_path / "config.yaml", "keep me")

    result = clear_workspace(tmp_path, config, include_input=True, include_generated=True)

    assert result.removed_count >= 7
    assert (tmp_path / "config.yaml").exists()
    for folder in ["input", "output", "temp", "logs", "data", "configs"]:
        remaining_files = [
            path
            for path in (tmp_path / folder).rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ]
        assert remaining_files == []


def test_clear_workspace_dry_run_keeps_files(tmp_path: Path) -> None:
    config = default_config()
    ensure_runtime_dirs(tmp_path, config)

    _write(tmp_path / "input" / "a.mp4")
    result = clear_workspace(tmp_path, config, include_input=True, include_generated=True, dry_run=True)

    assert result.dry_run is True
    assert (tmp_path / "input" / "a.mp4").exists()


def test_clear_workspace_preserves_google_drive_oauth_token(tmp_path: Path) -> None:
    config = default_config()
    config.storage.google_drive.oauth_token_file = "data/google-drive-token.json"
    ensure_runtime_dirs(tmp_path, config)

    _write(tmp_path / "data" / "google-drive-token.json", '{"token":"keep"}')
    _write(tmp_path / "data" / "jobs.json", "{}")

    clear_workspace(tmp_path, config, include_input=True, include_generated=True)

    assert (tmp_path / "data" / "google-drive-token.json").read_text(encoding="utf-8") == '{"token":"keep"}'
    assert not (tmp_path / "data" / "jobs.json").exists()
