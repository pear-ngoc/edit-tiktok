from __future__ import annotations

import json
import threading
from pathlib import Path

from config import default_config
from integrations.revid_api import (
    download_video_from_url,
    fetch_tiktok_download_info,
    select_download_url,
)
from integrations.telegram_bot import TelegramBotService
from integrations.tiktok import extract_tiktok_urls
from models import JobSource, JobStatus, ProcessResult, VideoJob
from queue_manager import QueueManager, discover_queueable_videos
from utils.files import VIDEO_EXTENSIONS


def _make_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    return path


def test_queue_scheduling_uses_limited_workers(tmp_path: Path) -> None:
    config = default_config()
    config.queue.max_workers = 5

    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    for index in range(10):
        _make_video(input_root / f"clip_{index}.mp4")

    active = 0
    max_active = 0
    started = 0
    release = threading.Event()
    first_wave = threading.Event()
    lock = threading.Lock()

    def process(job: VideoJob, progress_callback=None, **kwargs) -> ProcessResult:
        nonlocal active, max_active, started
        with lock:
            active += 1
            started += 1
            max_active = max(max_active, active)
            if active >= 5:
                first_wave.set()
        first_wave.wait(timeout=5)
        release.wait(timeout=5)
        output_path = Path(job.output_path or output_root / "fallback.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("done", encoding="utf-8")
        with lock:
            active -= 1
        return ProcessResult(Path(job.input_path), output_path, True, 0.01)

    manager = QueueManager(tmp_path, config, process)
    manager.start(watch_input=False, worker_count=5)
    assert first_wave.wait(timeout=5)
    assert max_active == 5
    release.set()
    manager.wait_for_idle()
    manager.stop()
    manager.join(timeout=5)
    assert started == 10
    assert max_active == 5


def test_input_watcher_discovers_supported_files_and_ignores_runtime_dirs(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _make_video(input_root / "clip.mp4")
    _make_video(input_root / "nested" / "clip.mov")
    _make_video(input_root / "output" / "skip.mp4")
    _make_video(input_root / "data" / "skip.mp4")
    _make_video(input_root / "failed" / "skip.mp4")
    _make_video(input_root / "completed" / "skip.mp4")
    _make_video(input_root / "logs" / "skip.mkv")
    (input_root / "notes.txt").write_text("hello", encoding="utf-8")

    videos = discover_queueable_videos(input_root, recursive=True)
    assert [path.name for path in videos] == ["clip.mp4", "clip.mov"]
    assert all(path.suffix.lower() in VIDEO_EXTENSIONS for path in videos)


def test_stable_file_check_accepts_stable_file(tmp_path: Path) -> None:
    config = default_config()
    manager = QueueManager(
        tmp_path,
        config,
        lambda job, progress_callback=None, **kwargs: ProcessResult(Path(job.input_path), None, True, 0.0),
    )
    video = _make_video(tmp_path / "input" / "stable.mp4")
    assert manager._wait_for_stable_file(video, 0.1) is True


def test_tiktok_url_extraction() -> None:
    text = """
    https://www.tiktok.com/@user/video/123
    https://vt.tiktok.com/abc
    https://www.tiktok.com/@user/video/123
    https://example.com/not-tiktok
    """
    urls = extract_tiktok_urls(text)
    assert urls == [
        "https://www.tiktok.com/@user/video/123",
        "https://vt.tiktok.com/abc",
    ]


def test_revid_response_parsing_and_fallback(monkeypatch, tmp_path: Path) -> None:
    payload = [
        {
            "video_url": "https://example.com/video.mp4",
            "download_direct": "https://example.com/direct.mp4",
            "uploader": "tester",
        }
    ]

    class FakeResponse:
        def __init__(self, body: bytes) -> None:
            self.body = body

        def read(self, *_args, **_kwargs):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=0):  # noqa: ANN001
        if request.full_url.endswith(".mp4"):
            return FakeResponse(b"video-bytes")
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("integrations.revid_api.urlopen", fake_urlopen)
    info = fetch_tiktok_download_info("https://www.tiktok.com/@a/video/1", "key", "https://api.test", 10)
    assert info[0]["video_url"] == "https://example.com/video.mp4"
    assert select_download_url(info) == "https://example.com/video.mp4"
    output = download_video_from_url("https://example.com/video.mp4", tmp_path / "video.mp4", 10)
    assert output.read_bytes() == b"video-bytes"


def test_revid_download_direct_fallback() -> None:
    payload = [{"download_direct": "https://example.com/direct.mp4"}]
    assert select_download_url(payload) == "https://example.com/direct.mp4"


def test_telegram_access_control_and_completion_delivery(tmp_path: Path, monkeypatch) -> None:
    config = default_config()
    config.telegram.allowed_chat_ids = [111, 222]
    config.telegram.allow_all_chats_if_empty = True
    dummy_qm = type("QM", (), {"update_job": lambda *args, **kwargs: None})()
    service = TelegramBotService(tmp_path, config, dummy_qm)

    assert service._is_chat_allowed(111) is True
    assert service._is_chat_allowed(333) is False

    config.telegram.allowed_chat_ids = []
    config.telegram.allow_all_chats_if_empty = True
    service = TelegramBotService(tmp_path, config, dummy_qm)
    assert service._is_chat_allowed(333) is True

    sent: dict[str, object] = {}

    def fake_send_document(chat_id: int, file_path: Path, *, caption: str) -> bool:
        sent["chat_id"] = chat_id
        sent["path"] = file_path
        sent["caption"] = caption
        return True

    monkeypatch.setattr(service, "_send_document", fake_send_document)
    monkeypatch.setattr(service, "_send_message", lambda *args, **kwargs: None)
    output = tmp_path / "output.mp4"
    output.write_bytes(b"x")
    job = VideoJob(
        job_id="job_123",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.COMPLETED,
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(output),
        chat_id=999,
        original_url="https://www.tiktok.com/@user/video/1",
    )
    result = ProcessResult(Path(job.input_path), output, True, 1.2)
    service.on_job_completed(job, result)

    assert sent["chat_id"] == 999
    assert sent["path"] == output
    assert "https://www.tiktok.com/@user/video/1" in str(sent["caption"])
