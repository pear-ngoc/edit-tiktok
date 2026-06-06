from __future__ import annotations

from pathlib import Path

from models import AppConfig


REQUIRED_DIRS = [
    "input",
    "input/telegram",
    "output",
    "assets/luts",
    "assets/font",
    "assets/ambient",
    "assets/bgm",
    "assets/overlays",
    "data",
    "secrets",
    "failed",
    "completed",
    "configs",
    "logs",
    "temp",
]


def project_root_from_file(file_path: str) -> Path:
    return Path(file_path).resolve().parent


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def ensure_runtime_dirs(project_root: Path, config: AppConfig | None = None) -> None:
    for rel in REQUIRED_DIRS:
        (project_root / rel).mkdir(parents=True, exist_ok=True)

    if config is not None:
        resolve_project_path(project_root, config.processing.input_dir).mkdir(parents=True, exist_ok=True)
        resolve_project_path(project_root, config.processing.output_dir).mkdir(parents=True, exist_ok=True)
        resolve_project_path(project_root, config.processing.temp_dir).mkdir(parents=True, exist_ok=True)
        resolve_project_path(project_root, config.audio.ambient_dir).mkdir(parents=True, exist_ok=True)
        resolve_project_path(project_root, config.audio.bgm_dir).mkdir(parents=True, exist_ok=True)
        resolve_project_path(project_root, config.subtitles.output_dir).mkdir(parents=True, exist_ok=True)
        resolve_project_path(project_root, config.queue.state_file).parent.mkdir(parents=True, exist_ok=True)
        resolve_project_path(project_root, config.storage.state_file).parent.mkdir(parents=True, exist_ok=True)
        resolve_project_path(project_root, config.telegram.input_subdir).mkdir(parents=True, exist_ok=True)
        if config.queue.move_failed_to:
            resolve_project_path(project_root, config.queue.move_failed_to).mkdir(parents=True, exist_ok=True)
        if config.queue.move_completed_to:
            resolve_project_path(project_root, config.queue.move_completed_to).mkdir(parents=True, exist_ok=True)


def ensure_gitkeep_files(project_root: Path) -> None:
    for rel in REQUIRED_DIRS:
        directory = project_root / rel
        directory.mkdir(parents=True, exist_ok=True)
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")
