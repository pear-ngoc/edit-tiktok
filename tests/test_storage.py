from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from config import config_from_dict, default_config
from models import JobSource, JobStatus, VideoJob
from storage.base import StorageContext, StorageUploadResult
from storage.google_drive import _resolve_credentials_path
from storage.manager import StorageManager
from storage.telegram import TelegramStorageProvider


def _job(tmp_path: Path, *, source: JobSource = JobSource.LOCAL_INPUT, chat_id: int | None = None) -> VideoJob:
    output = tmp_path / "output.mp4"
    return VideoJob(
        job_id="job_123abc",
        source=source,
        status=JobStatus.RENDERED,
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(output),
        chat_id=chat_id,
        identity="identity",
    )


def _video(tmp_path: Path, name: str = "output.mp4", size: int = 1024) -> Path:
    path = tmp_path / name
    path.write_bytes(b"x" * size)
    return path


def test_valid_storage_provider_parsing() -> None:
    for provider in ("local", "telegram", "google_drive", "both"):
        config = config_from_dict({"storage": {"provider": provider}})
        assert config.storage.provider == provider


def test_unknown_storage_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="storage.provider"):
        config_from_dict({"storage": {"provider": "ftp"}})


def test_google_credentials_path_resolves_from_project_root(tmp_path: Path) -> None:
    resolved = _resolve_credentials_path(tmp_path, "secrets/google-drive-service-account.json")
    assert resolved == tmp_path / "secrets/google-drive-service-account.json"


def test_telegram_job_uses_originating_chat_id(tmp_path: Path) -> None:
    config = default_config()
    config.storage.telegram.default_chat_id = 222
    sent: list[int] = []

    class Sender:
        def send_storage_document(self, chat_id: int, file_path: Path, *, caption: str) -> bool:
            sent.append(chat_id)
            return True

        def update_storage_status(self, job_id: str, text: str) -> bool:
            return True

    file_path = _video(tmp_path)
    job = _job(tmp_path, source=JobSource.TELEGRAM_TIKTOK, chat_id=111)
    result = TelegramStorageProvider(Sender()).upload(file_path, StorageContext(tmp_path, job, config))
    assert result.success is True
    assert sent == [111]


def test_local_job_uses_default_telegram_chat_id(tmp_path: Path) -> None:
    config = default_config()
    config.storage.telegram.default_chat_id = 333
    sent: list[int] = []

    class Sender:
        def send_storage_document(self, chat_id: int, file_path: Path, *, caption: str) -> bool:
            sent.append(chat_id)
            return True

        def update_storage_status(self, job_id: str, text: str) -> bool:
            return True

    result = TelegramStorageProvider(Sender()).upload(_video(tmp_path), StorageContext(tmp_path, _job(tmp_path), config))
    assert result.success is True
    assert sent == [333]


def test_missing_local_telegram_chat_id_returns_clear_failure(tmp_path: Path) -> None:
    config = default_config()
    result = TelegramStorageProvider(object()).upload(_video(tmp_path), StorageContext(tmp_path, _job(tmp_path), config))  # type: ignore[arg-type]
    assert result.success is False
    assert result.permanent is True
    assert "default_chat_id" in (result.error or "")


def test_oversized_telegram_file_does_not_retry(tmp_path: Path) -> None:
    config = default_config()
    config.storage.telegram.default_chat_id = 1
    config.storage.telegram.max_file_size_mb = 1

    class Sender:
        def send_storage_document(self, chat_id: int, file_path: Path, *, caption: str) -> bool:
            raise AssertionError("oversized files should not be sent")

        def update_storage_status(self, job_id: str, text: str) -> bool:
            return True

    result = TelegramStorageProvider(Sender()).upload(
        _video(tmp_path, size=2 * 1024 * 1024),
        StorageContext(tmp_path, _job(tmp_path), config),
    )
    assert result.success is False
    assert result.permanent is True


def test_storage_manager_rejects_zero_byte_output(tmp_path: Path) -> None:
    config = default_config()
    config.storage.provider = "telegram"
    path = _video(tmp_path, size=0)
    manager = StorageManager(tmp_path, config)
    results = manager.upload_final_output(path, _job(tmp_path), config)
    assert results[0].success is False
    assert "rỗng" in (results[0].error or "")


def test_selects_burned_output_over_non_burned_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = default_config()
    config.subtitles.burn_in = True
    normal = _video(tmp_path, "video.mp4")
    burned = _video(tmp_path, "video_burned.mp4")
    monkeypatch.setattr(StorageManager, "validate_final_output", lambda self, file_path: None)
    assert StorageManager(tmp_path, config).select_final_output(normal, config) == burned


def test_local_provider_performs_no_remote_upload(tmp_path: Path) -> None:
    config = default_config()
    config.storage.provider = "local"
    path = _video(tmp_path)
    manager = StorageManager(tmp_path, config)
    assert manager.upload_final_output(path, _job(tmp_path), config) == []


def test_successful_upload_is_persisted_and_not_duplicated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = default_config()
    config.storage.provider = "telegram"
    calls = {"count": 0}

    class Provider:
        name = "telegram"

        def upload(self, file_path: Path, context: StorageContext) -> StorageUploadResult:
            calls["count"] += 1
            return StorageUploadResult("telegram", True, file_path, remote_id="chat", uploaded_bytes=file_path.stat().st_size)

    path = _video(tmp_path)
    monkeypatch.setattr(StorageManager, "validate_final_output", lambda self, file_path: None)
    monkeypatch.setattr(StorageManager, "_providers", lambda self, cfg: [Provider()])
    manager = StorageManager(tmp_path, config)
    assert manager.upload_final_output(path, _job(tmp_path), config)[0].success is True
    restarted = StorageManager(tmp_path, config)
    assert restarted.upload_final_output(path, _job(tmp_path), config)[0].success is True
    assert calls["count"] == 1


def test_provider_both_runs_independently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = default_config()
    config.storage.provider = "both"

    class Provider:
        def __init__(self, name: str, success: bool) -> None:
            self.name = name
            self.success = success

        def upload(self, file_path: Path, context: StorageContext) -> StorageUploadResult:
            return StorageUploadResult(self.name, self.success, file_path, error=None if self.success else "failed")

    monkeypatch.setattr(StorageManager, "validate_final_output", lambda self, file_path: None)
    monkeypatch.setattr(StorageManager, "_providers", lambda self, cfg: [Provider("telegram", True), Provider("google_drive", False)])
    results = StorageManager(tmp_path, config).upload_final_output(_video(tmp_path), _job(tmp_path), config)
    assert [(item.provider, item.success) for item in results] == [("telegram", True), ("google_drive", False)]


def test_local_deletion_only_after_all_required_uploads_succeed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = default_config()
    config.storage.provider = "both"
    config.storage.delete_local_after_upload = True

    class Provider:
        def __init__(self, name: str, success: bool) -> None:
            self.name = name
            self.success = success

        def upload(self, file_path: Path, context: StorageContext) -> StorageUploadResult:
            return StorageUploadResult(self.name, self.success, file_path)

    path = _video(tmp_path)
    monkeypatch.setattr(StorageManager, "validate_final_output", lambda self, file_path: None)
    monkeypatch.setattr(StorageManager, "_providers", lambda self, cfg: [Provider("telegram", True), Provider("google_drive", False)])
    StorageManager(tmp_path, config).upload_final_output(path, _job(tmp_path), config)
    assert path.exists()


def test_storage_doctor_does_not_upload_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = default_config()
    config.storage.provider = "local"
    called = {"upload": False}

    def fail_upload(*args, **kwargs):  # noqa: ANN002, ANN003
        called["upload"] = True
        raise AssertionError("doctor must not upload")

    monkeypatch.setattr(StorageManager, "upload_final_output", fail_upload)
    lines = StorageManager(tmp_path, config).doctor()
    assert lines[0] == "Storage provider: local"
    assert called["upload"] is False
