from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path

from models import AppConfig, PreflightBatchResult, ProcessResult
from processing.pipeline import process_video
from processing.lut import resolve_lut_selection
from utils.files import find_video_files
from utils.runtime_logging import build_job_runtime_context, job_context_scope, log_runtime_execution_plan, resolve_whisper_runtime

LOGGER = logging.getLogger(__name__)


def run_batch(
    project_root: Path,
    config: AppConfig,
    *,
    preflight_result: PreflightBatchResult | None = None,
) -> list[ProcessResult]:
    input_root = _resolve(project_root, config.processing.input_dir)
    output_root = _resolve(project_root, config.processing.output_dir)
    temp_root = _resolve(project_root, config.processing.temp_dir)
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    if preflight_result is not None:
        videos = [result.source for result in preflight_result.processable_videos]
    else:
        videos = find_video_files(input_root, recursive=config.processing.recursive)

    if not videos:
        print(f"Không tìm thấy video nào trong {input_root}")
        LOGGER.info("Không tìm thấy video nào trong %s", input_root)
        return []

    lut_result = resolve_lut_selection(project_root, config.color)
    lut_paths = list(lut_result.resolved_paths)
    for warning in lut_result.warnings:
        print(warning)

    if lut_paths:
        print("Đang áp dụng LUT: " + ", ".join(path.name for path in lut_paths))

    workers = choose_worker_count(config.processing.max_workers)
    print(f"Tìm thấy {len(videos)} video. Đang xử lý với {workers} luồng.")
    LOGGER.info("Bắt đầu xử lý hàng loạt: %s video, %s luồng", len(videos), workers)

    results: list[ProcessResult] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_video,
                video,
                input_root=input_root,
                output_root=output_root,
                temp_root=temp_root,
                project_root=project_root,
                config=config,
                lut_paths=lut_paths,
                worker_slot=index + 1,
                worker_total=workers,
            )
            for index, video in enumerate(videos)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            status = "THÀNH CÔNG" if result.success else "LỖI"
            target = result.output if result.output else result.source
            print(f"[{completed}/{len(videos)}] {status}: {target}")

    success_count = sum(1 for result in results if result.success)
    print(f"Hoàn tất: {success_count}/{len(results)} video thành công.")
    return results


def choose_worker_count(value: int | str) -> int:
    if isinstance(value, int):
        return max(1, value)
    if str(value).lower() != "auto":
        try:
            return max(1, int(value))
        except ValueError:
            return 1
    cpu_count = os.cpu_count() or 2
    return max(1, min(2, cpu_count // 2))


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
