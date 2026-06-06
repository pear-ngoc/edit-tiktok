from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from config import default_config
from integrations.telegram_bot import TelegramBotService
from models import JobSource, JobStatus, ProcessResult, VideoJob


class DummyQueueManager:
    def __init__(self) -> None:
        self.jobs: dict[str, VideoJob] = {}
        self.enqueue_calls: list[dict[str, object]] = []
        self.update_calls: list[tuple[str, dict[str, object]]] = []
        self.clear_state_called = False

    def enqueue_path(self, path: Path, **kwargs):  # noqa: ANN001
        self.enqueue_calls.append({"path": path, **kwargs})
        job = VideoJob(
            job_id=f"job_{len(self.enqueue_calls)}",
            source=kwargs.get("source", JobSource.TELEGRAM_TIKTOK),
            status=JobStatus.QUEUED,
            input_path=str(path),
            output_path=str(path.with_suffix(".mp4")),
            chat_id=kwargs.get("chat_id"),
            telegram_chat_id=kwargs.get("telegram_chat_id"),
            telegram_status_message_id=kwargs.get("telegram_status_message_id"),
            telegram_status_text=kwargs.get("telegram_status_text", ""),
            original_url=kwargs.get("original_url"),
        )
        self.jobs[job.job_id] = job
        return job

    def load_job(self, job_id: str) -> VideoJob | None:
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **changes: object) -> VideoJob | None:
        self.update_calls.append((job_id, changes))
        job = self.jobs.get(job_id)
        if job is None:
            return None
        for key, value in changes.items():
            if hasattr(job, key):
                setattr(job, key, value)
        return job

    def clear_state(self) -> None:
        self.clear_state_called = True


def _make_service(tmp_path: Path, monkeypatch, config=None) -> tuple[TelegramBotService, DummyQueueManager]:
    cfg = config or default_config()
    cfg.telegram.send_progress_messages = True
    cfg.telegram.edit_progress_message = True
    cfg.telegram.bot_token = "token"
    queue_manager = DummyQueueManager()
    service = TelegramBotService(tmp_path, cfg, queue_manager)
    service._application = SimpleNamespace(
        bot=SimpleNamespace(
            send_message=lambda *args, **kwargs: None,
            edit_message_text=lambda *args, **kwargs: None,
            send_document=lambda *args, **kwargs: None,
        )
    )
    service._loop = object()
    return service, queue_manager


def test_initial_link_creates_one_status_message_and_stores_ids(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    sent = {"count": 0}

    class FakeMessage:
        message_id = 777

    async def fake_send_initial_status_message(chat_id: int, text: str):  # noqa: ANN001
        sent["count"] += 1
        sent["chat_id"] = chat_id
        sent["text"] = text
        return FakeMessage()

    monkeypatch.setattr(service, "_send_initial_status_message", fake_send_initial_status_message)
    monkeypatch.setattr(
        "integrations.telegram_bot.fetch_tiktok_download_info",
        lambda *args, **kwargs: [{"video_url": "https://example.com/video.mp4", "uploader": "tester"}],
    )
    monkeypatch.setattr("integrations.telegram_bot.select_download_url", lambda payload: payload[0]["video_url"])
    monkeypatch.setattr(
        "integrations.telegram_bot.download_video_from_url",
        lambda url, output_path, timeout: output_path.write_bytes(b"video") or output_path,
    )

    async def run() -> None:
        await service._handle_tiktok_url(SimpleNamespace(), 7032252869, "https://www.tiktok.com/@user/video/1")

    asyncio.run(run())

    assert sent["count"] == 1
    assert "đang tải xuống" in sent["text"].lower()
    assert queue_manager.enqueue_calls[0]["telegram_chat_id"] == 7032252869
    assert queue_manager.enqueue_calls[0]["telegram_status_message_id"] == 777
    assert queue_manager.enqueue_calls[0]["telegram_status_text"] == sent["text"]
    assert queue_manager.enqueue_calls[0]["original_url"] == "https://www.tiktok.com/@user/video/1"


def test_status_text_edit_is_deduplicated(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    job = VideoJob(
        job_id="job_1",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.QUEUED,
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(tmp_path / "output.mp4"),
        chat_id=111,
        telegram_chat_id=111,
        telegram_status_message_id=55,
        telegram_status_text="Đang xử lý",
    )
    queue_manager.jobs[job.job_id] = job
    edits: list[str] = []

    async def fake_edit_message_text(chat_id: int, message_id: int, text: str) -> bool:  # noqa: ANN001
        edits.append(text)
        return True

    monkeypatch.setattr(service, "_edit_message_text", fake_edit_message_text)

    async def run() -> None:
        assert await service.update_job_status_message(job, "Đang xử lý") is True
        assert await service.update_job_status_message(job, "🎬 Đang xử lý video...") is True

    asyncio.run(run())

    assert edits == ["🎬 Đang xử lý video..."]
    assert queue_manager.update_calls[-1][1]["telegram_status_text"] == "🎬 Đang xử lý video..."


def test_same_message_used_for_queue_processing_and_subtitles(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    job = VideoJob(
        job_id="job_2",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.QUEUED,
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(tmp_path / "output.mp4"),
        chat_id=222,
        telegram_chat_id=222,
        telegram_status_message_id=88,
        telegram_status_text="⏳ Đã nhận link TikTok, đang tải xuống... https://www.tiktok.com/@user/video/1",
    )
    queue_manager.jobs[job.job_id] = job
    edits: list[str] = []

    monkeypatch.setattr(service, "_edit_status_message_sync", lambda _job, text: edits.append(text) or True)

    service.on_job_queued(job)
    service.on_job_started(job)
    service.on_job_stage(job, "generating_subtitles", "📝 Đang tạo phụ đề...")
    service.on_job_stage(job, "burning_subtitles", "🔥 Đang burn phụ đề vào video...")

    assert edits == [
        "✅ Tải xuống hoàn tất\n🕒 Video đang chờ trong hàng đợi...",
        "🎬 Đang xử lý video...\nTên file: input.mp4",
        "📝 Đang tạo phụ đề...",
        "🔥 Đang burn phụ đề vào video...",
    ]


def test_storage_send_uses_one_document_and_existing_runtime(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    output = tmp_path / "output.mp4"
    output.write_bytes(b"x" * 1024)
    job = VideoJob(
        job_id="job_3",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.PROCESSING,
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(output),
        chat_id=333,
        telegram_chat_id=333,
        telegram_status_message_id=99,
        telegram_status_text="Đang xử lý",
        original_url="https://www.tiktok.com/@user/video/1",
    )
    queue_manager.jobs[job.job_id] = job
    edits: list[str] = []
    sent: list[tuple[int, Path, str]] = []

    monkeypatch.setattr(service, "_edit_status_message_sync", lambda _job, text: edits.append(text) or True)
    monkeypatch.setattr(service, "_send_document", lambda chat_id, file_path, *, caption: sent.append((chat_id, file_path, caption)) or True)

    assert service.update_storage_status(job.job_id, "📤 Đang tải output lên Telegram...") is True
    assert service.send_storage_document(333, output, caption="caption") is True

    assert sent == [(333, output, "caption")]
    assert edits == ["📤 Đang tải output lên Telegram..."]


def test_failure_edits_original_status_without_sending_new_message(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    job = VideoJob(
        job_id="job_4",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.PROCESSING,
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(tmp_path / "output.mp4"),
        chat_id=444,
        telegram_chat_id=444,
        telegram_status_message_id=100,
        telegram_status_text="Đang xử lý",
    )
    queue_manager.jobs[job.job_id] = job
    edits: list[str] = []
    monkeypatch.setattr(service, "_edit_status_message_sync", lambda _job, text: edits.append(text) or True)
    monkeypatch.setattr(service, "_send_document", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not send")))

    service.on_job_failed(job, "traceback here")

    assert len(edits) == 1
    assert edits[0].startswith("❌ Xử lý thất bại")


def test_completion_status_edit_failure_does_not_fail_processing_job(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    output = tmp_path / "output.mp4"
    output.write_bytes(b"x")
    job = VideoJob(
        job_id="job_5",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.PROCESSING,
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(output),
        chat_id=555,
        telegram_chat_id=555,
        telegram_status_message_id=101,
        telegram_status_text="Đang xử lý",
    )
    queue_manager.jobs[job.job_id] = job
    edits: list[str] = []
    monkeypatch.setattr(service, "_edit_status_message_sync", lambda _job, text: edits.append(text) or False)
    sent: list[object] = []
    monkeypatch.setattr(service, "_send_document", lambda chat_id, file_path, *, caption: sent.append((chat_id, file_path, caption)) or True)

    service.on_job_completed(job, ProcessResult(Path(job.input_path), output, True, 1.0))

    assert sent == []
    assert edits and edits[0].startswith("✅ Hoàn tất")


def test_worker_thread_bridge_uses_run_coroutine_threadsafe(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    job = VideoJob(
        job_id="job_6",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.PROCESSING,
        input_path=str(tmp_path / "input.mp4"),
        output_path=str(tmp_path / "output.mp4"),
        chat_id=666,
        telegram_chat_id=666,
        telegram_status_message_id=102,
        telegram_status_text="Đang xử lý",
    )
    queue_manager.jobs[job.job_id] = job
    called: dict[str, object] = {}

    class FakeFuture:
        def result(self, timeout=None):  # noqa: ANN001
            called["timeout"] = timeout
            return True

    def fake_run_coroutine_threadsafe(coro, loop):  # noqa: ANN001
        called["coro"] = coro
        called["loop"] = loop
        return FakeFuture()

    monkeypatch.setattr("integrations.telegram_bot.asyncio.run_coroutine_threadsafe", fake_run_coroutine_threadsafe)
    async def fake_update_job_status_message(_job, _text):  # noqa: ANN001
        return True

    monkeypatch.setattr(service, "update_job_status_message", fake_update_job_status_message)

    assert service._edit_status_message_sync(job, "🎬 Đang xử lý video...") is True
    assert called["loop"] is service._loop
    assert "coro" in called


def test_telegram_clear_command_runs_without_confirmation(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_clear(project_root, config, *, include_input, include_generated, dry_run):  # noqa: ANN001
        calls.append(
            {
                "project_root": project_root,
                "include_input": include_input,
                "include_generated": include_generated,
                "dry_run": dry_run,
            }
        )
        return SimpleNamespace(removed_count=7)

    monkeypatch.setattr("integrations.telegram_bot.clear_runtime_workspace", fake_clear)
    replies: list[str] = []

    class FakeMessage:
        async def reply_text(self, text: str) -> None:
            replies.append(text)

    async def run() -> None:
        await service._handle_clear_request(FakeMessage(), 123, scope="all")

    asyncio.run(run())

    assert calls == [
        {
            "project_root": tmp_path,
            "include_input": True,
            "include_generated": True,
            "dry_run": False,
        }
    ]
    assert queue_manager.clear_state_called is True
    assert replies[-1].startswith("✅ Đã clear workspace.")


def test_two_jobs_keep_separate_status_message_ids(tmp_path: Path, monkeypatch) -> None:
    service, queue_manager = _make_service(tmp_path, monkeypatch)
    edits: list[tuple[int, int, str]] = []

    async def fake_edit_message_text(chat_id: int, message_id: int, text: str) -> bool:  # noqa: ANN001
        edits.append((chat_id, message_id, text))
        return True

    monkeypatch.setattr(service, "_edit_message_text", fake_edit_message_text)

    job_a = VideoJob(
        job_id="job_a",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.QUEUED,
        input_path=str(tmp_path / "a.mp4"),
        output_path=str(tmp_path / "a_out.mp4"),
        chat_id=7001,
        telegram_chat_id=7001,
        telegram_status_message_id=201,
        telegram_status_text="A",
    )
    job_b = VideoJob(
        job_id="job_b",
        source=JobSource.TELEGRAM_TIKTOK,
        status=JobStatus.QUEUED,
        input_path=str(tmp_path / "b.mp4"),
        output_path=str(tmp_path / "b_out.mp4"),
        chat_id=7002,
        telegram_chat_id=7002,
        telegram_status_message_id=202,
        telegram_status_text="B",
    )
    queue_manager.jobs[job_a.job_id] = job_a
    queue_manager.jobs[job_b.job_id] = job_b

    async def run() -> None:
        assert await service.update_job_status_message(job_a, "A-1") is True
        assert await service.update_job_status_message(job_b, "B-1") is True

    asyncio.run(run())

    assert edits == [(7001, 201, "A-1"), (7002, 202, "B-1")]
