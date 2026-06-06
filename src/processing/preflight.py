from __future__ import annotations

from pathlib import Path

from models import AppConfig, PreflightBatchResult, PreflightStatus, PreflightVideoResult
from utils.files import find_video_files
from utils.paths import resolve_project_path


def discover_input_videos(project_root: Path, config: AppConfig) -> list[Path]:
    input_root = resolve_project_path(project_root, config.processing.input_dir)
    return find_video_files(input_root, recursive=config.processing.recursive)


def ensure_input_videos_exist(project_root: Path, config: AppConfig) -> list[Path]:
    videos = discover_input_videos(project_root, config)
    if videos:
        return videos
    print("No input videos found. Please add videos to the input/ folder and run again.")
    return []


def run_preflight_checks(project_root: Path, config: AppConfig) -> PreflightBatchResult:
    input_root = resolve_project_path(project_root, config.processing.input_dir)
    videos = discover_input_videos(project_root, config)
    return PreflightBatchResult(
        input_root=input_root,
        videos=[
            PreflightVideoResult(
                source=video,
                status=PreflightStatus.VALID,
                supported=True,
            )
            for video in videos
        ],
    )


def print_preflight_summary(result: PreflightBatchResult) -> None:
    print("Preflight check:")
    print(f"* Input folder: {result.input_root}/")
    print(f"* Videos found: {result.total_count}")

    if result.total_count == 0:
        print("\nNo input videos found. Please add videos to the input/ folder and run again.")
        return

    print(f"* Valid: {result.total_count}")
    print("\nProcessing will continue.")
