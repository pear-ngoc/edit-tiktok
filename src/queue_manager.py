from __future__ import annotations

import hashlib
import json
import logging
import queue as queue_module
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from models import AppConfig, JobSource, JobStatus, ProcessResult, VideoJob
from storage import StorageManager, StorageUploadResult
from utils.files import find_video_files, safe_output_path, sanitize_filename
from utils.paths import resolve_project_path
from utils.runtime_logging import build_job_runtime_context, job_context_scope

LOGGER = logging.getLogger(__name__)

_WATCH_EXCLUDED_DIRS = {"output", "temp", "logs", "data", "failed", "completed"}


class QueueEventSink(Protocol):
    def on_job_queued(self, job: VideoJob) -> None: ...

    def on_job_started(self, job: VideoJob) -> None: ...

    def on_job_stage(self, job: VideoJob, stage: str, text: str) -> None: ...

    def on_job_completed(self, job: VideoJob, result: ProcessResult) -> None: ...

    def on_job_failed(self, job: VideoJob, error: str) -> None: ...


ProgressCallback = Callable[[str, str], None]
ProcessCallback = Callable[..., ProcessResult]


class QueueManager:
    def __init__(
        self,
        project_root: Path,
        config: AppConfig,
        process_callback: ProcessCallback,
        *,
        notifier: QueueEventSink | None = None,
        storage_manager: StorageManager | None = None,
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.process_callback = process_callback
        self.notifier = notifier
        self.storage_manager = storage_manager
        self.input_root = resolve_project_path(project_root, config.processing.input_dir)
        self.output_root = resolve_project_path(project_root, config.processing.output_dir)
        self.temp_root = resolve_project_path(project_root, config.processing.temp_dir)
        self.state_path = resolve_project_path(project_root, config.queue.state_file)
        self.downloads_dir = project_root / "data" / "downloads"
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, VideoJob] = {}
        self._jobs_by_identity: dict[str, str] = {}
        self._jobs_by_path: dict[str, str] = {}
        self._pending_queue: queue_module.Queue[str | None] = queue_module.Queue()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._watcher: threading.Thread | None = None
        self._started = False
        self._last_heartbeat = 0.0
        self._load_state()

    @property
    def jobs(self) -> list[VideoJob]:
        with self._lock:
            return list(self._jobs.values())

    def start(self, *, watch_input: bool = False, worker_count: int | None = None) -> None:
        if self._started:
            return
        self._started = True
        count = max(1, worker_count or self.config.queue.max_workers)
        LOGGER.info("Khởi động queue manager | workers=%s | watch_input=%s", count, watch_input)
        self._log_queue_snapshot(prefix="QUEUE")
        for index in range(count):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(index + 1, count),
                name=f"edit-tiktok-worker-{index + 1}",
                daemon=True,
            )
            thread.start()
            self._workers.append(thread)

        if watch_input:
            self._watcher = threading.Thread(
                target=self._watch_loop,
                name="edit-tiktok-watcher",
                daemon=True,
            )
            self._watcher.start()
        else:
            self.scan_and_enqueue_existing()

    def stop(self) -> None:
        if self._stop_event.is_set():
            return
        LOGGER.info("Đang yêu cầu dừng queue manager")
        self._stop_event.set()
        if self._watcher and self._watcher.is_alive():
            self._watcher.join(timeout=5)
        for _ in self._workers:
            self._pending_queue.put(None)
        self.save_state()

    def join(self, timeout: float | None = None) -> None:
        start = time.monotonic()
        for worker in self._workers:
            remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - start))
            worker.join(timeout=remaining)
        if self._watcher and self._watcher.is_alive():
            remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - start))
            self._watcher.join(timeout=remaining)

    def wait_for_idle(self) -> None:
        while not self._stop_event.is_set():
            if self._pending_queue.empty() and not self._has_active_processing():
                return
            time.sleep(0.2)

    def scan_and_enqueue_existing(self) -> int:
        videos = discover_queueable_videos(self.input_root, recursive=self.config.processing.recursive)
        queued = 0
        for path in videos:
            if self.enqueue_path(path, source=JobSource.LOCAL_INPUT, queue_now=True) is not None:
                queued += 1
        LOGGER.info("Quét input xong | found=%s | queued=%s", len(videos), queued)
        return queued

    def enqueue_path(
        self,
        path: Path,
        *,
        source: JobSource,
        chat_id: int | None = None,
        telegram_chat_id: int | None = None,
        telegram_status_message_id: int | None = None,
        telegram_status_text: str = "",
        original_url: str | None = None,
        queue_now: bool = True,
    ) -> VideoJob | None:
        resolved = path.resolve()
        if not resolved.exists() or not resolved.is_file():
            LOGGER.warning("Bỏ qua file không tồn tại: %s", resolved)
            return None
        if not self._is_queueable_path(resolved):
            LOGGER.info("Bỏ qua file ngoài phạm vi theo dõi: %s", resolved)
            return None

        stat = resolved.stat()
        identity = build_job_identity(resolved, stat.st_size, stat.st_mtime)
        with self._lock:
            existing = self._jobs_by_identity.get(identity)
            if existing:
                job = self._jobs[existing]
                if chat_id is not None and job.chat_id is None:
                    job.chat_id = chat_id
                if telegram_chat_id is not None and job.telegram_chat_id is None:
                    job.telegram_chat_id = telegram_chat_id
                if telegram_status_message_id is not None and job.telegram_status_message_id is None:
                    job.telegram_status_message_id = telegram_status_message_id
                if telegram_status_text and not job.telegram_status_text:
                    job.telegram_status_text = telegram_status_text
                if original_url and not job.original_url:
                    job.original_url = original_url
                if source == JobSource.TELEGRAM_TIKTOK and job.source == JobSource.LOCAL_INPUT:
                    job.source = source
                self.save_state_locked()
                duplicate_level = getattr(logging, self.config.logging.queue_duplicate_log_level.upper(), logging.DEBUG)
                LOGGER.log(
                    duplicate_level,
                    "Bỏ qua queue trùng lặp | job_id=%s | status=%s | input=%s",
                    job.job_id,
                    job.status.value,
                    resolved,
                )
                return job

            job_id = build_job_id(identity)
            output_path = safe_output_path(resolved, self.input_root, self.output_root)
            job = VideoJob(
                job_id=job_id,
                source=source,
                status=JobStatus.QUEUED if queue_now else JobStatus.PENDING,
                input_path=str(resolved),
                output_path=str(output_path),
                chat_id=chat_id,
                telegram_chat_id=telegram_chat_id if telegram_chat_id is not None else chat_id,
                telegram_status_message_id=telegram_status_message_id,
                telegram_status_text=telegram_status_text,
                original_url=original_url,
                created_at=utc_now_iso(),
                file_size=stat.st_size,
                modified_time=stat.st_mtime,
                identity=identity,
            )
            self._jobs[job_id] = job
            self._jobs_by_identity[identity] = job_id
            self._jobs_by_path[str(resolved)] = job_id
            if queue_now:
                self._pending_queue.put(job_id)
            self.save_state_locked()

        LOGGER.info(
            "Đã tạo job | job_id=%s | source=%s | input=%s | output=%s | chat_id=%s | status=%s",
            job.job_id,
            job.source.value,
            job.input_path,
            job.output_path,
            job.chat_id,
            job.status.value,
        )
        if self.notifier and queue_now:
            self._safe_notify("on_job_queued", job)
        return job

    def update_job(self, job_id: str, **changes: Any) -> VideoJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            for key, value in changes.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            self.save_state_locked()
            return job

    def load_job(self, job_id: str) -> VideoJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def completed_jobs_recent(self, limit: int = 20) -> list[VideoJob]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.status == JobStatus.COMPLETED]
            jobs.sort(key=lambda item: item.completed_at or item.created_at, reverse=True)
            return jobs[:limit]

    def _worker_loop(self, worker_slot: int, worker_total: int) -> None:
        LOGGER.info("Worker bắt đầu: %s", threading.current_thread().name)
        while True:
            if self._stop_event.is_set() and self._pending_queue.empty():
                break
            try:
                item = self._pending_queue.get(timeout=0.5)
            except queue_module.Empty:
                continue
            if item is None:
                self._pending_queue.task_done()
                break
            job = self.load_job(item)
            if job is None:
                self._pending_queue.task_done()
                continue
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}:
                self._pending_queue.task_done()
                continue

            self._log_worker_assignment(worker_slot, worker_total, job)
            self._set_status(job.job_id, JobStatus.PROCESSING, started_at=utc_now_iso())
            self._safe_notify("on_job_started", job)
            job_context = build_job_runtime_context(
                job_id=job.job_id,
                source=job.source.value,
                input_path=Path(job.input_path),
                output_path=Path(job.output_path) if job.output_path else None,
                worker_slot=worker_slot,
                worker_total=worker_total,
            )
            LOGGER.info(
                "Bắt đầu xử lý job | job_id=%s | source=%s | input=%s | output=%s",
                job.job_id,
                job.source.value,
                job.input_path,
                job.output_path,
            )
            try:
                with job_context_scope(
                    job_context,
                    log_dir=self.project_root / "logs" / "jobs",
                    enabled=getattr(self.config.logging, "per_job_logs", True),
                    level=logging.getLogger().level,
                ):
                    try:
                        def progress(stage: str, text: str) -> None:
                            self._safe_notify("on_job_stage", self.load_job(job.job_id) or job, stage, text)

                        result = self.process_callback(
                            job,
                            progress,
                            worker_slot=worker_slot,
                            worker_total=worker_total,
                        )
                        if result.output and result.output.exists():
                            self._set_status(
                                job.job_id,
                                JobStatus.RENDERED,
                                output_path=str(result.output),
                                output_size=result.output.stat().st_size,
                                error=None,
                            )
                            storage_results = self._deliver_rendered_output(job.job_id, result.output)
                            final_status = (
                                JobStatus.COMPLETED
                                if all(item.success for item in storage_results)
                                else JobStatus.UPLOAD_FAILED
                            )
                            self._set_status(
                                job.job_id,
                                final_status,
                                completed_at=utc_now_iso(),
                                error=None if final_status == JobStatus.COMPLETED else _storage_error_summary(storage_results),
                            )
                        elif result.success:
                            self._set_status(
                                job.job_id,
                                JobStatus.COMPLETED,
                                completed_at=utc_now_iso(),
                                error=None,
                            )
                        else:
                            raise RuntimeError(result.error or "Job thất bại không rõ lý do")

                        LOGGER.info(
                            "Hoàn tất job | job_id=%s | output=%s | success=%s | elapsed=%.2fs",
                            job.job_id,
                            result.output or job.output_path,
                            result.success,
                            result.elapsed_seconds,
                        )
                        self._log_queue_snapshot(prefix=f"WORKER {worker_slot}/{worker_total}")
                        current_job = self.load_job(job.job_id) or job
                        if current_job.status == JobStatus.COMPLETED:
                            self._safe_notify("on_job_completed", current_job, result)
                    except Exception as exc:  # pragma: no cover - safety net
                        LOGGER.exception("Job thất bại: %s", job.job_id)
                        self._set_status(
                            job.job_id,
                            JobStatus.FAILED,
                            completed_at=utc_now_iso(),
                            error=str(exc),
                        )
                        self._log_queue_snapshot(prefix=f"WORKER {worker_slot}/{worker_total}")
                        self._safe_notify("on_job_failed", self.load_job(job.job_id) or job, str(exc))
            finally:
                self._pending_queue.task_done()

        LOGGER.info("Worker dừng: %s", threading.current_thread().name)

    def _watch_loop(self) -> None:
        LOGGER.info("Watcher bắt đầu | input=%s", self.input_root)
        self.scan_and_enqueue_existing()
        interval = max(0.5, float(self.config.queue.scan_interval_seconds))
        stable_seconds = max(0.5, float(self.config.queue.stable_file_check_seconds))
        heartbeat_seconds = max(5.0, float(self.config.logging.queue_heartbeat_seconds))
        while not self._stop_event.wait(interval):
            videos = discover_queueable_videos(self.input_root, recursive=self.config.processing.recursive)
            queued_now = 0
            for path in videos:
                if self._stop_event.is_set():
                    break
                if not self._wait_for_stable_file(path, stable_seconds):
                    continue
                if self.enqueue_path(path, source=JobSource.LOCAL_INPUT, queue_now=True) is not None:
                    queued_now += 1
            now = time.monotonic()
            if now - self._last_heartbeat >= heartbeat_seconds:
                self._last_heartbeat = now
                snapshot = self._queue_snapshot()
                LOGGER.info(
                    "[WATCHER] Running | active=%s | waiting=%s | completed=%s | failed=%s | found=%s | queued=%s",
                    snapshot["active"],
                    snapshot["waiting"],
                    snapshot["completed"],
                    snapshot["failed"],
                    len(videos),
                    queued_now,
                )
        LOGGER.info("Watcher dừng")

    def _wait_for_stable_file(self, path: Path, stable_seconds: float) -> bool:
        try:
            first = path.stat()
        except FileNotFoundError:
            return False
        start = time.monotonic()
        last_size = first.st_size
        last_mtime = first.st_mtime
        while time.monotonic() - start < stable_seconds:
            time.sleep(min(0.5, stable_seconds))
            try:
                current = path.stat()
            except FileNotFoundError:
                return False
            if current.st_size != last_size or current.st_mtime != last_mtime:
                last_size = current.st_size
                last_mtime = current.st_mtime
                start = time.monotonic()
        return True

    def _has_active_processing(self) -> bool:
        with self._lock:
            return any(job.status in {JobStatus.PROCESSING, JobStatus.UPLOADING} for job in self._jobs.values())

    def _deliver_rendered_output(self, job_id: str, output_path: Path) -> list[StorageUploadResult]:
        job = self.load_job(job_id)
        if job is None:
            return []
        if self.config.storage.provider == "local":
            return []
        if self.storage_manager is None:
            return [
                StorageUploadResult(
                    provider=self.config.storage.provider,
                    success=False,
                    local_path=output_path,
                    error="Storage manager chưa được cấu hình.",
                    permanent=True,
                )
            ]
        self._set_status(job_id, JobStatus.UPLOADING)
        self._safe_notify("on_job_stage", job, "uploading", _storage_start_text(self.config.storage.provider))
        results = self.storage_manager.upload_final_output(output_path, job, self.config)
        self._safe_notify("on_job_stage", self.load_job(job_id) or job, "uploading", _storage_final_text(results))
        return results

    def _set_status(self, job_id: str, status: JobStatus, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = status
            for key, value in changes.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            self.save_state_locked()

    def _safe_notify(self, method_name: str, *args: Any) -> None:
        if self.notifier is None:
            return
        method = getattr(self.notifier, method_name, None)
        if method is None:
            return
        try:
            method(*args)
        except Exception:  # pragma: no cover - logging only
            LOGGER.exception("Lỗi khi gửi thông báo queue: %s", method_name)

    def _log_queue_snapshot(self, *, prefix: str) -> None:
        snapshot = self._queue_snapshot()
        LOGGER.info(
            "[%s] max_workers=%s active=%s waiting=%s completed=%s failed=%s",
            prefix,
            self.config.queue.max_workers,
            snapshot["active"],
            snapshot["waiting"],
            snapshot["completed"],
            snapshot["failed"],
        )

    def _log_worker_assignment(self, worker_slot: int, worker_total: int, job: VideoJob) -> None:
        snapshot = self._queue_snapshot()
        LOGGER.info(
            "[WORKER %s/%s] Assigned job %s: %s | source=%s | active=%s waiting=%s completed=%s failed=%s",
            worker_slot,
            worker_total,
            job.job_id,
            Path(job.input_path).name,
            job.source.value,
            snapshot["active"],
            snapshot["waiting"],
            snapshot["completed"],
            snapshot["failed"],
        )

    def _queue_snapshot(self) -> dict[str, int]:
        with self._lock:
            active = sum(1 for job in self._jobs.values() if job.status in {JobStatus.PROCESSING, JobStatus.UPLOADING})
            waiting = sum(1 for job in self._jobs.values() if job.status in {JobStatus.PENDING, JobStatus.QUEUED})
            completed = sum(1 for job in self._jobs.values() if job.status == JobStatus.COMPLETED)
            failed = sum(1 for job in self._jobs.values() if job.status in {JobStatus.FAILED, JobStatus.UPLOAD_FAILED})
        return {"active": active, "waiting": waiting, "completed": completed, "failed": failed}

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.exception("Không đọc được file trạng thái queue: %s", self.state_path)
            return
        jobs = raw.get("jobs", []) if isinstance(raw, dict) else []
        for item in jobs:
            try:
                job = job_from_dict(item)
            except Exception:
                LOGGER.exception("Bỏ qua job trạng thái không hợp lệ: %s", item)
                continue
            self._jobs[job.job_id] = job
            self._jobs_by_identity[job.identity] = job.job_id
            self._jobs_by_path[job.input_path] = job.job_id
        LOGGER.info("Đã tải trạng thái queue | jobs=%s", len(self._jobs))

    def save_state(self) -> None:
        with self._lock:
            self.save_state_locked()

    def save_state_locked(self) -> None:
        payload = {
            "updated_at": utc_now_iso(),
            "jobs": [job_to_dict(job) for job in self._jobs.values()],
        }
        tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.state_path)

    def _is_queueable_path(self, path: Path) -> bool:
        try:
            relative = path.resolve().relative_to(self.input_root.resolve())
        except ValueError:
            return False
        parts = {part.lower() for part in relative.parts[:-1]}
        if parts & _WATCH_EXCLUDED_DIRS:
            return False
        return path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def discover_queueable_videos(input_root: Path, recursive: bool = True) -> list[Path]:
    videos = find_video_files(input_root, recursive=recursive)
    return [path for path in videos if not _contains_excluded_folder(path, input_root)]


def _contains_excluded_folder(path: Path, input_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(input_root.resolve())
    except ValueError:
        return True
    return any(part.lower() in _WATCH_EXCLUDED_DIRS for part in relative.parts[:-1])


def build_job_identity(path: Path, size: int, modified_time: float) -> str:
    absolute = path.resolve().as_posix()
    return f"{absolute}|{size}|{modified_time:.6f}"


def build_job_id(identity: str) -> str:
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()
    return f"job_{digest[:12]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_to_dict(job: VideoJob) -> dict[str, Any]:
    data = asdict(job)
    data["source"] = job.source.value
    data["status"] = job.status.value
    return data


def job_from_dict(data: dict[str, Any]) -> VideoJob:
    return VideoJob(
        job_id=str(data["job_id"]),
        source=JobSource(str(data["source"])),
        status=JobStatus(str(data["status"])),
        input_path=str(data["input_path"]),
        output_path=str(data["output_path"]) if data.get("output_path") else None,
        chat_id=int(data["chat_id"]) if data.get("chat_id") is not None else None,
        telegram_chat_id=int(data["telegram_chat_id"]) if data.get("telegram_chat_id") is not None else None,
        telegram_status_message_id=int(data["telegram_status_message_id"]) if data.get("telegram_status_message_id") is not None else None,
        telegram_status_text=str(data.get("telegram_status_text", "")),
        original_url=str(data["original_url"]) if data.get("original_url") else None,
        created_at=str(data.get("created_at", "")),
        started_at=str(data["started_at"]) if data.get("started_at") else None,
        completed_at=str(data["completed_at"]) if data.get("completed_at") else None,
        error=str(data["error"]) if data.get("error") else None,
        file_size=int(data.get("file_size", 0)),
        modified_time=float(data.get("modified_time", 0.0)),
        identity=str(data.get("identity", "")),
        metadata_path=str(data["metadata_path"]) if data.get("metadata_path") else None,
        output_size=int(data.get("output_size", 0)),
    )


def sanitize_download_filename(uploader: str, job_id: str) -> str:
    safe_uploader = sanitize_filename(uploader or "tiktok")
    return f"tiktok_{safe_uploader}_{job_id}.mp4"


def _storage_start_text(provider: str) -> str:
    if provider == "telegram":
        return "📤 Đang tải output lên Telegram..."
    if provider == "google_drive":
        return "☁️ Đang tải output lên Google Drive..."
    if provider == "both":
        return "📤 Đang gửi video lên Telegram...\n☁️ Đang tải bản sao lên Google Drive..."
    return ""


def _storage_final_text(results: list[StorageUploadResult]) -> str:
    if not results:
        return "✅ Hoàn tất"
    lines = ["✅ Upload hoàn tất" if all(item.success for item in results) else "⚠️ Render hoàn tất", ""]
    for item in results:
        label = "Telegram" if item.provider == "telegram" else "Google Drive" if item.provider == "google_drive" else item.provider
        lines.append(f"{label}: {'đã upload' if item.success else 'upload thất bại'}")
        if item.provider == "google_drive" and item.remote_url:
            lines.append(f"Drive: {item.remote_url}")
    if not all(item.success for item in results):
        lines.append("Video local vẫn được giữ lại.")
    return "\n".join(lines)


def _storage_error_summary(results: list[StorageUploadResult]) -> str:
    failed = [f"{item.provider}: {item.error or 'upload failed'}" for item in results if not item.success]
    return "; ".join(failed) or None
