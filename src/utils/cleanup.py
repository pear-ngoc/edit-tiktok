from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from models import AppConfig
from utils.paths import ensure_gitkeep_files, ensure_runtime_dirs, resolve_project_path


@dataclass(slots=True)
class CleanupResult:
    removed_paths: list[Path]
    dry_run: bool = False

    @property
    def removed_count(self) -> int:
        return len(self.removed_paths)


def build_clear_targets(
    project_root: Path,
    config: AppConfig,
    *,
    include_input: bool = True,
    include_generated: bool = True,
) -> list[Path]:
    targets: list[Path] = []
    if include_input:
        targets.append(resolve_project_path(project_root, config.processing.input_dir))
    if include_generated:
        targets.extend(
            [
                resolve_project_path(project_root, config.processing.output_dir),
                resolve_project_path(project_root, config.processing.temp_dir),
                project_root / "logs",
                project_root / "data",
                project_root / "failed",
                project_root / "completed",
                project_root / "configs",
            ]
        )
    unique_targets: list[Path] = []
    seen: set[Path] = set()
    for target in targets:
        resolved = target.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_targets.append(resolved)
    return unique_targets


def clear_workspace(
    project_root: Path,
    config: AppConfig,
    *,
    include_input: bool = True,
    include_generated: bool = True,
    dry_run: bool = False,
) -> CleanupResult:
    preserved_files = _read_preserved_files(project_root, config)
    preserved_dirs = _move_preserved_dirs(project_root, config) if not dry_run else {}
    targets = build_clear_targets(
        project_root,
        config,
        include_input=include_input,
        include_generated=include_generated,
    )
    removed: list[Path] = []
    for target in targets:
        if not target.exists():
            continue
        removed.append(target)
        if dry_run:
            continue
        shutil.rmtree(target, ignore_errors=True)

    if not dry_run:
        ensure_gitkeep_files(project_root)
        ensure_runtime_dirs(project_root, config)
        _restore_preserved_files(preserved_files)
        _restore_preserved_dirs(preserved_dirs)

    return CleanupResult(removed_paths=removed, dry_run=dry_run)


def _read_preserved_files(project_root: Path, config: AppConfig) -> dict[Path, bytes]:
    paths = [
        resolve_project_path(project_root, config.storage.google_drive.oauth_token_file),
    ]
    preserved: dict[Path, bytes] = {}
    for path in paths:
        if path.exists() and path.is_file():
            preserved[path] = path.read_bytes()
    return preserved


def _restore_preserved_files(files: dict[Path, bytes]) -> None:
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _move_preserved_dirs(project_root: Path, config: AppConfig) -> dict[Path, Path]:
    paths = [
        project_root / "data" / "huggingface",
    ]
    preserved: dict[Path, Path] = {}
    preserve_root = project_root / ".clear-preserve"
    for path in paths:
        if not path.exists() or not path.is_dir():
            continue
        relative = path.resolve().relative_to(project_root.resolve())
        target = preserve_root / relative
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        preserved[path] = target
    return preserved


def _restore_preserved_dirs(dirs: dict[Path, Path]) -> None:
    for original, preserved in dirs.items():
        if not preserved.exists():
            continue
        if original.exists():
            shutil.rmtree(original, ignore_errors=True)
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(preserved), str(original))
    for preserved in set(dirs.values()):
        root = preserved
        while root.name and root.name != ".clear-preserve":
            root = root.parent
        if root.name == ".clear-preserve" and root.exists():
            shutil.rmtree(root, ignore_errors=True)
