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

    return CleanupResult(removed_paths=removed, dry_run=dry_run)
