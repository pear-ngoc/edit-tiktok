from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ffmpeg_tools.probe import probe_video
from models import AppConfig, JobSource, JobStatus, VideoJob
from storage.base import StorageContext, StorageProvider, StorageUploadResult
from storage.google_drive import GoogleDriveStorageProvider
from storage.local import LocalStorageProvider
from storage.telegram import DirectTelegramSender, TelegramSender, TelegramStorageProvider
from utils.paths import resolve_project_path

LOGGER = logging.getLogger(__name__)


class StorageManager:
    def __init__(
        self,
        project_root: Path,
        config: AppConfig,
        *,
        telegram_sender: TelegramSender | None = None,
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.state_path = resolve_project_path(project_root, config.storage.state_file)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.telegram_sender = telegram_sender
        if self.telegram_sender is None and config.storage.provider in {"telegram", "both"} and config.telegram.bot_token.strip():
            self.telegram_sender = DirectTelegramSender(config)
        self._state_lock = threading.RLock()
        self._upload_semaphore = threading.Semaphore(max(1, int(config.storage.max_concurrent_uploads)))
        self._state = self._load_state()

    def upload_final_output(
        self,
        file_path: Path,
        job: VideoJob,
        config: AppConfig | None = None,
    ) -> list[StorageUploadResult]:
        cfg = config or self.config
        providers = self._providers(cfg)
        if not providers:
            return []
        final_path = self.select_final_output(Path(file_path), cfg)
        LOGGER.info("STORAGE | %s | provider=%s", job.job_id, cfg.storage.provider)
        LOGGER.info(
            "STORAGE | %s | final=%s | size=%.1fMB",
            job.job_id,
            final_path.name,
            final_path.stat().st_size / (1024 * 1024) if final_path.exists() else 0,
        )
        validation_error = self.validate_final_output(final_path)
        if validation_error:
            return [
                StorageUploadResult(
                    provider=provider.name,
                    success=False,
                    local_path=final_path,
                    error=validation_error,
                    permanent=True,
                )
                for provider in providers
            ]

        context = StorageContext(project_root=self.project_root, job=job, config=cfg)
        results: list[StorageUploadResult] = []
        with self._upload_semaphore:
            for provider in providers:
                result = self._upload_with_state(provider, final_path, context)
                results.append(result)
            if cfg.storage.upload_subtitles:
                for subtitle_path in self._subtitle_files(final_path, cfg):
                    subtitle_context = StorageContext(
                        project_root=self.project_root,
                        job=job,
                        config=cfg,
                        is_subtitle=True,
                    )
                    for provider in providers:
                        results.append(self._upload_with_state(provider, subtitle_path, subtitle_context))
        if cfg.storage.delete_local_after_upload and all(result.success for result in results):
            final_path.unlink(missing_ok=True)
            LOGGER.info("STORAGE | %s | local output deleted after upload", job.job_id)
        elif not all(result.success for result in results):
            LOGGER.info("STORAGE | %s | local output preserved", job.job_id)
        return results

    def upload_existing_file(self, file_path: Path, job: VideoJob | None = None) -> list[StorageUploadResult]:
        upload_job = job or VideoJob(
            job_id=build_upload_job_id(file_path),
            source=JobSource.LOCAL_INPUT,
            status=JobStatus.RENDERED,
            input_path=str(file_path),
            output_path=str(file_path),
        )
        return self.upload_final_output(file_path, upload_job, self.config)

    def retry_failed_uploads(self) -> list[StorageUploadResult]:
        results: list[StorageUploadResult] = []
        with self._state_lock:
            failed_items = [
                item for item in self._state.get("uploads", {}).values()
                if item.get("status") in {"failed", "pending"}
            ]
        for item in failed_items:
            path = Path(str(item.get("local_path", "")))
            if not path.exists():
                continue
            job = VideoJob(
                job_id=str(item.get("job_id") or build_upload_job_id(path)),
                source=JobSource.LOCAL_INPUT,
                status=JobStatus.RENDERED,
                input_path=str(path),
                output_path=str(path),
            )
            provider_name = str(item.get("provider", ""))
            provider = self._provider_by_name(provider_name, self.config)
            if provider is None:
                continue
            results.append(self._upload_with_state(provider, path, StorageContext(self.project_root, job, self.config)))
        return results

    def status_lines(self) -> list[str]:
        with self._state_lock:
            uploads = list(self._state.get("uploads", {}).values())
        if not uploads:
            return ["Chưa có upload nào."]
        lines = []
        for item in sorted(uploads, key=lambda value: str(value.get("completed_at") or value.get("updated_at") or "")):
            lines.append(
                f"{item.get('job_id')} | {item.get('provider')} | {item.get('status')} | "
                f"{Path(str(item.get('local_path', ''))).name} | {item.get('remote_url') or item.get('remote_id') or ''}"
            )
        return lines

    def validate_final_output(self, file_path: Path) -> str | None:
        if not file_path.exists():
            return "Final output không tồn tại."
        if not file_path.is_file():
            return "Final output không phải file."
        if file_path.stat().st_size <= 0:
            return "Final output rỗng."
        try:
            info = probe_video(file_path)
        except Exception as exc:
            return f"ffprobe không đọc được final output: {exc}"
        if info.duration <= 0 or info.width <= 0 or info.height <= 0:
            return "Final output không có stream video hợp lệ."
        return None

    def select_final_output(self, output_path: Path, config: AppConfig) -> Path:
        if not config.subtitles.burn_in or output_path.stem.endswith("_burned"):
            return output_path
        burned = output_path.with_name(f"{output_path.stem}_burned{output_path.suffix}")
        if burned.exists() and self.validate_final_output(burned) is None:
            return burned
        candidates = sorted(output_path.parent.glob(f"{output_path.stem}_burned*{output_path.suffix}"))
        for candidate in candidates:
            if self.validate_final_output(candidate) is None:
                return candidate
        return output_path

    def _subtitle_files(self, final_path: Path, config: AppConfig) -> list[Path]:
        subtitle_dir = resolve_project_path(self.project_root, config.subtitles.output_dir)
        if not subtitle_dir.exists():
            return []
        stems = {final_path.stem}
        if final_path.stem.endswith("_burned"):
            stems.add(final_path.stem[: -len("_burned")])
        matches: list[Path] = []
        for suffix in (".srt", ".ass", ".vtt"):
            for stem in stems:
                candidate = subtitle_dir / f"{stem}{suffix}"
                if candidate.exists() and candidate.is_file():
                    matches.append(candidate)
        return sorted(set(matches))

    def doctor(self) -> list[str]:
        cfg = self.config.storage
        lines = [f"Storage provider: {cfg.provider}", "", "Telegram:"]
        telegram_configured = bool(self.config.telegram.bot_token.strip())
        lines.append(f"  configured: {'yes' if telegram_configured else 'no'}")
        lines.append(f"  connection: {'not checked' if not telegram_configured else 'configured'}")
        lines.append(f"  default chat: {'configured' if cfg.telegram.default_chat_id is not None else 'not configured'}")
        lines.extend(["", "Google Drive:"])
        auth_method = (cfg.google_drive.auth_method or "service_account").strip().lower()
        credentials = _drive_auth_path(self.project_root, cfg.google_drive)
        token_path = _drive_token_path(self.project_root, cfg.google_drive)
        drive_needed = cfg.provider in {"google_drive", "both"}
        lines.append(f"  auth method: {auth_method}")
        credential_label = "oauth client" if auth_method == "oauth" else "credentials"
        lines.append(f"  {credential_label}: {'found' if credentials.exists() else 'missing'}")
        if auth_method == "oauth":
            lines.append(f"  token: {'found' if token_path.exists() else 'missing'}")
        if not drive_needed:
            lines.append("  authentication: not required")
            lines.append("  folder access: not required")
        elif auth_method == "oauth" and not token_path.exists():
            lines.append("  authentication: missing token; run storage auth")
            lines.append("  folder access: not checked")
        elif credentials.exists() and cfg.google_drive.folder_id.strip():
            try:
                service = provider_module_build(self.project_root, cfg.google_drive)
                lines.append("  authentication: passed")
                provider_module_verify(service, cfg.google_drive.folder_id.strip(), cfg.google_drive.shared_drive_id.strip())
                lines.append("  folder access: passed")
            except Exception as exc:
                lines.append(f"  authentication: failed ({_safe_error(exc)})")
                lines.append("  folder access: failed")
        else:
            lines.append("  authentication: failed")
            lines.append("  folder access: failed")
        lines.append(f"  folder ID: {_mask(cfg.google_drive.folder_id)}")
        return lines

    def _providers(self, config: AppConfig) -> list[StorageProvider]:
        provider = config.storage.provider
        if provider == "local":
            return []
        if provider == "telegram":
            return [TelegramStorageProvider(self.telegram_sender)]
        if provider == "google_drive":
            return [GoogleDriveStorageProvider()]
        if provider == "both":
            return [TelegramStorageProvider(self.telegram_sender), GoogleDriveStorageProvider()]
        return []

    def _provider_by_name(self, provider_name: str, config: AppConfig) -> StorageProvider | None:
        for provider in [TelegramStorageProvider(self.telegram_sender), GoogleDriveStorageProvider(), LocalStorageProvider()]:
            if provider.name == provider_name:
                return provider
        return None

    def _upload_with_state(
        self,
        provider: StorageProvider,
        file_path: Path,
        context: StorageContext,
    ) -> StorageUploadResult:
        key = build_upload_key(context.job.job_id, provider.name, file_path)
        existing = self._state.get("uploads", {}).get(key)
        if existing and existing.get("status") == "completed":
            return StorageUploadResult(
                provider=provider.name,
                success=True,
                local_path=file_path,
                remote_id=existing.get("remote_id"),
                remote_url=existing.get("remote_url"),
                uploaded_bytes=int(existing.get("file_size", 0)),
            )
        attempts = max(1, int(context.config.storage.retry_attempts))
        delay = max(0.0, float(context.config.storage.retry_delay_seconds))
        last: StorageUploadResult | None = None
        for attempt in range(1, attempts + 1):
            self._record_attempt(key, provider.name, file_path, context.job, attempt, "pending")
            LOGGER.info("STORAGE | %s | %s | uploading", context.job.job_id, provider.name)
            last = provider.upload(file_path, context)
            if last.success or last.permanent:
                break
            if attempt < attempts:
                time.sleep(delay * (2 ** (attempt - 1)))
        result = last or StorageUploadResult(provider.name, False, file_path, error="Upload không chạy.")
        self._record_result(key, result, context.job)
        return result

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"uploads": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("uploads"), dict):
                return data
        except Exception:
            LOGGER.exception("Không đọc được storage state: %s", self.state_path)
        return {"uploads": {}}

    def _save_state(self) -> None:
        with self._state_lock:
            self._state["updated_at"] = utc_now_iso()
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)

    def _record_attempt(self, key: str, provider: str, file_path: Path, job: VideoJob, attempt: int, status: str) -> None:
        stat = file_path.stat()
        with self._state_lock:
            self._state.setdefault("uploads", {})[key] = {
                **self._state.setdefault("uploads", {}).get(key, {}),
                "job_id": job.job_id,
                "provider": provider,
                "local_path": str(file_path),
                "file_size": stat.st_size,
                "modified_time": stat.st_mtime,
                "status": status,
                "attempts": attempt,
                "updated_at": utc_now_iso(),
            }
        self._save_state()

    def _record_result(self, key: str, result: StorageUploadResult, job: VideoJob) -> None:
        with self._state_lock:
            item = self._state.setdefault("uploads", {}).setdefault(key, {})
            item.update(
                {
                    "job_id": job.job_id,
                    "provider": result.provider,
                    "local_path": str(result.local_path),
                    "status": "completed" if result.success else "failed",
                    "remote_id": result.remote_id,
                    "remote_url": result.remote_url,
                    "last_error": result.error,
                    "permanent": result.permanent,
                    "completed_at": utc_now_iso() if result.success else None,
                    "updated_at": utc_now_iso(),
                }
            )
        self._save_state()


def build_upload_key(job_id: str, provider: str, file_path: Path) -> str:
    stat = file_path.stat()
    raw = f"{job_id}|{provider}|{file_path.resolve()}|{stat.st_size}|{stat.st_mtime:.6f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_upload_job_id(file_path: Path) -> str:
    raw = str(file_path.resolve())
    return "upload_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mask(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


def _safe_error(exc: Exception) -> str:
    return " ".join(str(exc).split())[:160]


def provider_module_build(project_root: Path, cfg: Any) -> Any:
    from storage.google_drive import build_drive_service

    return build_drive_service(project_root, cfg)


def provider_module_verify(service: Any, folder_id: str, shared_drive_id: str) -> None:
    from storage.google_drive import _verify_folder_access

    _verify_folder_access(service, folder_id, shared_drive_id)


def _drive_auth_path(project_root: Path, cfg: Any) -> Path:
    method = (cfg.auth_method or "service_account").strip().lower()
    configured = cfg.oauth_client_secrets_file if method == "oauth" else cfg.credentials_file
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def _drive_token_path(project_root: Path, cfg: Any) -> Path:
    path = Path(cfg.oauth_token_file)
    return path if path.is_absolute() else project_root / path
