from __future__ import annotations

import asyncio
import math
import logging
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from integrations.revid_api import (
    download_video_from_url,
    fetch_tiktok_download_info,
    fetch_tiktokdl_info,
    select_download_url,
    select_tiktokdl_url,
)
from integrations.tiktok import extract_tiktok_urls, parse_tiktok_url_input
from models import AppConfig, JobSource, JobStatus, ProcessResult, VideoJob
from utils.files import sanitize_filename, unique_path
from utils.cleanup import clear_workspace as clear_runtime_workspace
from utils.paths import ensure_runtime_dirs, resolve_project_path
from utils.runtime_logging import build_job_runtime_context, job_context_scope, job_prefix

LOGGER = logging.getLogger(__name__)

_STAGE_TEXTS = {
    "downloading": lambda job, *_: f"⏳ Đã nhận link TikTok, đang tải xuống... {job.original_url or ''}".strip(),
    "downloading_revid": lambda job, *_: "⏳ Đang tải xuống (Revid API)...",
    "downloading_fallback": lambda job, *_: "⏳ Revid thất bại, thử tiktokdl...",
    "queued": lambda job, *_: "✅ Tải xuống hoàn tất\n🕒 Video đang chờ trong hàng đợi...",
    "processing": lambda job, *_: f"🎬 Đang xử lý video...\nTên file: {Path(job.input_path).name}",
    "generating_subtitles": lambda job, *_: "📝 Đang tạo phụ đề...",
    "generating_subtitles_groq": lambda job, *_: "📝 Đang tạo phụ đề bằng Groq...",
    "burning_subtitles": lambda job, *_: "🔥 Đang burn phụ đề vào video...",
    "fallback_to_local": lambda job, *_: "⚠️ Groq tạm thời không khả dụng\nĐang chuyển sang nhận dạng local...",
    "sending": lambda job, *_: "📤 Xử lý hoàn tất, đang gửi video...",
    "uploading_telegram": lambda job, *_: "📤 Đang tải output lên Telegram...",
    "uploading_drive": lambda job, *_: "☁️ Đang tải output lên Google Drive...",
    "completed": lambda job, output_path=None, *_: f"✅ Hoàn tất\n📁 {(output_path or Path(job.output_path or '')).name}",
    "failed": lambda job, _output_path=None, error=None: f"❌ Xử lý thất bại\nLỗi: {error or 'Không xác định'}",
}


@dataclass(slots=True)
class TelegramDeliveryResult:
    sent: bool
    message: str = ""


class TelegramBotService:
    TELEGRAM_OFFICIAL_UPLOAD_LIMIT_MB = 50

    def __init__(self, project_root: Path, config: AppConfig, queue_manager: Any) -> None:
        self.project_root = project_root
        self.config = config
        self.queue_manager = queue_manager
        self.telegram_config = config.telegram
        self.revid_config = config.revid_api
        self.tiktokdl_config = config.tiktokdl_fallback
        self.input_dir = resolve_project_path(project_root, self.telegram_config.input_subdir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        ensure_runtime_dirs(project_root, config)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._application: Any = None
        self._telegram_bad_request: type[BaseException] | tuple[type[BaseException], ...] = Exception
        self._telegram_retry_after: type[BaseException] | tuple[type[BaseException], ...] = Exception

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        token = self.telegram_config.bot_token.strip()
        if not token:
            raise RuntimeError("Thiếu Telegram bot token")
        self._import_telegram_modules()
        self._thread = threading.Thread(target=self._run_loop, name="edit-tiktok-telegram", daemon=True)
        self._thread.start()
        LOGGER.info("Telegram bot đang khởi động")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)
        LOGGER.info("Telegram bot đã dừng")

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def on_job_queued(self, job: VideoJob) -> None:
        if (
            not self.telegram_config.send_progress_messages
            or not self.telegram_config.edit_progress_message
            or job.chat_id is None
        ):
            return
        if job.source != JobSource.TELEGRAM_TIKTOK:
            return
        self._edit_status_message_sync(job, self._stage_text(job, "queued"))

    def on_job_started(self, job: VideoJob) -> None:
        if (
            not self.telegram_config.send_progress_messages
            or not self.telegram_config.edit_progress_message
            or job.chat_id is None
        ):
            return
        if job.source != JobSource.TELEGRAM_TIKTOK:
            return
        self._edit_status_message_sync(job, self._stage_text(job, "processing"))

    def on_job_stage(self, job: VideoJob, stage: str, text: str) -> None:
        if not self.telegram_config.send_progress_messages or job.chat_id is None:
            return
        if job.source != JobSource.TELEGRAM_TIKTOK:
            return
        if not self.telegram_config.edit_progress_message:
            return
        self._edit_status_message_sync(job, text or self._stage_text(job, stage))

    def on_job_completed(self, job: VideoJob, result: ProcessResult) -> None:
        if self.config.storage.provider != "local":
            return
        output_path = result.output or (Path(job.output_path) if job.output_path else None)
        if output_path and output_path.exists() and job.source == JobSource.TELEGRAM_TIKTOK:
            self._edit_status_message_sync(job, self._stage_text(job, "completed", output_path=output_path))

    def on_job_failed(self, job: VideoJob, error: str) -> None:
        if (
            job.chat_id is None
            or job.source != JobSource.TELEGRAM_TIKTOK
            or not self.telegram_config.edit_progress_message
        ):
            return
        self._edit_status_message_sync(job, self._stage_text(job, "failed", error=_safe_telegram_error(error)))
        LOGGER.error("Telegram job thất bại | chat_id=%s | job_id=%s | error=%s", job.chat_id, job.job_id, error)

    def _import_telegram_modules(self) -> None:
        try:
            import telegram  # noqa: F401
            from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
            from telegram.request import HTTPXRequest
            from telegram.error import BadRequest, RetryAfter
        except Exception as exc:  # pragma: no cover - dependency missing
            LOGGER.exception("Không nạp được python-telegram-bot")
            raise RuntimeError(
                "Thiếu phụ thuộc python-telegram-bot. Hãy cài để chạy Telegram bot."
            ) from exc
        self._telegram_application_builder = ApplicationBuilder
        self._telegram_command_handler = CommandHandler
        self._telegram_message_handler = MessageHandler
        self._telegram_filters = filters
        self._telegram_httpx_request_class = HTTPXRequest
        self._telegram_bad_request = BadRequest
        self._telegram_retry_after = RetryAfter

    def _run_loop(self) -> None:
        self._import_telegram_modules()

        token = self.telegram_config.bot_token.strip()
        request = self._telegram_httpx_request_class(
            connection_pool_size=16,
            read_timeout=30.0,
            write_timeout=30.0,
            connect_timeout=10.0,
            pool_timeout=5.0,
            media_write_timeout=600.0,
        )
        builder = self._telegram_application_builder().token(token).request(request)
        api_settings = self._telegram_api_settings()
        if api_settings["base_url"]:
            builder = builder.base_url(api_settings["base_url"])
        if api_settings["base_file_url"]:
            builder = builder.base_file_url(api_settings["base_file_url"])
        builder = builder.local_mode(api_settings["local_mode"])
        LOGGER.info(
            "Telegram Bot API | base_url=%s | base_file_url=%s | local_mode=%s",
            api_settings["base_url"] or "default",
            api_settings["base_file_url"] or "default",
            api_settings["local_mode"],
        )
        application = builder.build()
        application.add_handler(self._telegram_command_handler("clear", self._handle_clear_command))
        application.add_handler(
            self._telegram_message_handler(
                self._telegram_filters.TEXT & ~self._telegram_filters.COMMAND,
                self._handle_message,
            )
        )
        self._application = application
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        LOGGER.info("Telegram bot đang chạy long polling")
        try:
            self._loop.run_until_complete(self._run_async(application))
        finally:
            self._loop.close()

    async def _run_async(self, application: Any) -> None:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        try:
            await asyncio.to_thread(self._stop_event.wait)
        finally:
            await self._shutdown_async()

    async def _shutdown_async(self) -> None:
        if self._application is None:
            return
        try:
            await self._application.updater.stop()
        except Exception:
            LOGGER.exception("Lỗi khi dừng Telegram updater")
        try:
            await self._application.stop()
        except Exception:
            LOGGER.exception("Lỗi khi dừng Telegram application")
        try:
            await self._application.shutdown()
        except Exception:
            LOGGER.exception("Lỗi khi shutdown Telegram application")

    async def _handle_message(self, update: Any, context: Any) -> None:  # noqa: ANN401
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        chat_id = int(chat.id)
        allowed = self._is_chat_allowed(chat_id)
        LOGGER.info("Telegram nhận tin nhắn | chat_id=%s | allowed=%s", chat_id, allowed)
        if not allowed:
            if self.telegram_config.allow_all_chats_if_empty:
                return
            if message.text:
                await message.reply_text("Bạn không có quyền sử dụng bot này.")
            return

        text = message.text or message.caption or ""
        if text.strip().lower() == "clear":
            await self._handle_clear_request(message, chat_id, scope="all")
            return
        urls = extract_tiktok_urls(text)
        LOGGER.info("Telegram URLs trích xuất | chat_id=%s | count=%s | urls=%s", chat_id, len(urls), urls)
        if not urls:
            return

        for raw_url in urls:
            url, subtitle_language_override = parse_tiktok_url_input(raw_url)
            await self._handle_tiktok_url(message, chat_id, url, subtitle_language_override)

    async def _handle_clear_command(self, update: Any, context: Any) -> None:  # noqa: ANN401
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None:
            return
        chat_id = int(chat.id)
        if not self._is_chat_allowed(chat_id):
            await message.reply_text("Bạn không có quyền sử dụng bot này.")
            return
        args = list(getattr(context, "args", []) or [])
        scope = args[0].strip().lower() if args else "all"
        await self._handle_clear_request(message, chat_id, scope=scope)

    async def _handle_clear_request(self, message: Any, chat_id: int, *, scope: str = "all") -> None:  # noqa: ANN401
        if scope not in {"all", "input", "generated"}:
            await message.reply_text("Scope clear không hợp lệ. Dùng: /clear, /clear input, hoặc /clear generated.")
            return
        include_input = scope in {"all", "input"}
        include_generated = scope in {"all", "generated"}
        await message.reply_text("🧹 Đang clear workspace...")
        try:
            result = await asyncio.to_thread(
                clear_runtime_workspace,
                self.project_root,
                self.config,
                include_input=include_input,
                include_generated=include_generated,
                dry_run=False,
            )
            clear_state = getattr(self.queue_manager, "clear_state", None)
            if callable(clear_state):
                await asyncio.to_thread(clear_state)
            LOGGER.info("Telegram clear workspace | chat_id=%s | scope=%s | removed=%s", chat_id, scope, result.removed_count)
            await message.reply_text(
                "✅ Đã clear workspace.\n"
                f"Scope: {scope}\n"
                f"Đã xoá: {result.removed_count} thư mục\n"
                "Google Drive OAuth token được giữ lại."
            )
        except Exception as exc:
            LOGGER.exception("Telegram clear workspace thất bại | chat_id=%s | scope=%s", chat_id, scope)
            await message.reply_text(f"❌ Clear thất bại: {_safe_telegram_error(str(exc))}")

    async def _handle_tiktok_url(
        self,
        message: Any,
        chat_id: int,
        url: str,
        subtitle_language_override: str | None = None,
    ) -> None:  # noqa: ANN401
        try:
            initial_message = None
            initial_text = self._stage_text(
                VideoJob(
                    job_id="pending",
                    source=JobSource.TELEGRAM_TIKTOK,
                    status=JobStatus.PENDING,
                    input_path="",
                    original_url=url,
                    telegram_chat_id=chat_id,
                    chat_id=chat_id,
                ),
                "downloading",
            )
            if self.telegram_config.send_progress_messages:
                initial_message = await self._send_initial_status_message(chat_id, initial_text)
            # Try Revid API first
            payload = await asyncio.to_thread(
                fetch_tiktok_download_info,
                url,
                self.revid_config.api_key,
                self.revid_config.endpoint,
                self.revid_config.timeout_seconds,
            )
            download_url = select_download_url(payload)
            uploader = str(payload[0].get("uploader") or "tiktok")
            timeout_seconds = self.revid_config.download_timeout_seconds

            if self.tiktokdl_config.enabled:
                try:
                    target_path = unique_path(self.input_dir / f"tiktok_{uploader}.mp4")
                    await asyncio.to_thread(download_video_from_url, download_url, target_path, timeout_seconds)
                except Exception:
                    LOGGER.warning("Revid download thất bại, thử tiktokdl cho %s", url)
                    if initial_message is not None:
                        await self._edit_message_text(
                            chat_id=chat_id,
                            message_id=initial_message.message_id,
                            text=self._stage_text(
                                VideoJob(
                                    job_id="pending",
                                    source=JobSource.TELEGRAM_TIKTOK,
                                    status=JobStatus.PENDING,
                                    input_path="",
                                    original_url=url,
                                    telegram_chat_id=chat_id,
                                    chat_id=chat_id,
                                ),
                                "downloading_fallback",
                            ),
                        )
                    tiktokdl_data = await asyncio.to_thread(
                        fetch_tiktokdl_info,
                        url,
                        self.tiktokdl_config.endpoint,
                        self.tiktokdl_config.tkdl_nonce,
                        self.tiktokdl_config.timeout_seconds,
                    )
                    download_url = select_tiktokdl_url(tiktokdl_data)
                    uploader = str(
                        tiktokdl_data.get("author", {}).get("unique_id", "tiktok")
                        if isinstance(tiktokdl_data.get("author"), dict)
                        else "tiktok"
                    )
                    timeout_seconds = self.tiktokdl_config.download_timeout_seconds
                    target_path = unique_path(self.input_dir / f"tiktok_{uploader}.mp4")
                    await asyncio.to_thread(download_video_from_url, download_url, target_path, timeout_seconds)
                    payload = tiktokdl_data
                    LOGGER.info("Đã tải TikTok bằng tiktokdl fallback | chat_id=%s | url=%s", chat_id, url)
            else:
                target_path = unique_path(self.input_dir / f"tiktok_{uploader}.mp4")
                await asyncio.to_thread(download_video_from_url, download_url, target_path, timeout_seconds)
            LOGGER.info(
                "Đã tải TikTok về input/telegram | chat_id=%s | url=%s | path=%s",
                chat_id,
                url,
                target_path,
            )
            job = self.queue_manager.enqueue_path(
                target_path,
                source=JobSource.TELEGRAM_TIKTOK,
                chat_id=chat_id,
                telegram_chat_id=chat_id,
                telegram_status_message_id=initial_message.message_id if initial_message else None,
                telegram_status_text=initial_text if initial_message else "",
                original_url=url,
                subtitle_language_override=subtitle_language_override,
                queue_now=True,
            )
            if job is None:
                if initial_message is not None:
                    await self._edit_message_text(
                        chat_id=chat_id,
                        message_id=initial_message.message_id,
                        text=self._stage_text(
                            VideoJob(
                                job_id="pending",
                                source=JobSource.TELEGRAM_TIKTOK,
                                status=JobStatus.PENDING,
                                input_path="",
                                original_url=url,
                                telegram_chat_id=chat_id,
                                chat_id=chat_id,
                            ),
                            "failed",
                            error="Không thể đưa video vào hàng đợi.",
                        ),
                    )
                return
            self._save_download_metadata(job, payload, download_url)
            LOGGER.info("Đã đưa job Telegram vào queue | chat_id=%s | job_id=%s", chat_id, job.job_id)
        except Exception as exc:
            LOGGER.exception("Lỗi khi xử lý link TikTok")
            LOGGER.error("Telegram xử lý link thất bại | chat_id=%s | url=%s | error=%s", chat_id, url, _safe_telegram_error(str(exc)))
            if initial_message is not None:
                await self._edit_message_text(
                    chat_id=chat_id,
                    message_id=initial_message.message_id,
                    text=self._stage_text(
                        VideoJob(
                            job_id="pending",
                            source=JobSource.TELEGRAM_TIKTOK,
                            status=JobStatus.PENDING,
                            input_path="",
                            original_url=url,
                            telegram_chat_id=chat_id,
                            chat_id=chat_id,
                        ),
                        "failed",
                        error=_safe_telegram_error(str(exc)),
                    ),
                )

    def _save_download_metadata(
        self,
        job: VideoJob,
        payload: list[dict[str, object]] | dict[str, object],
        download_url: str,
    ) -> None:
        metadata_dir = self.project_root / "data" / "downloads"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = metadata_dir / f"{job.job_id}.json"
        data = {
            "job_id": job.job_id,
            "chat_id": job.chat_id,
            "original_url": job.original_url,
            "download_url": download_url,
            "payload": payload,
        }
        metadata_path.write_text(json_dumps(data), encoding="utf-8")
        self.queue_manager.update_job(job.job_id, metadata_path=str(metadata_path))

    def _build_caption(self, job: VideoJob, output_path: Path) -> str:
        pieces = [self.telegram_config.message_when_done, f"File: {output_path.name}"]
        if job.original_url:
            pieces.append(f"Link: {job.original_url}")
        pieces.append(f"Job: {job.job_id}")
        return "\n".join(pieces)

    async def _send_initial_status_message(self, chat_id: int, text: str) -> Any:
        if self._application is None:
            return None
        try:
            return await self._application.bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            LOGGER.exception("Không gửi được tin nhắn trạng thái ban đầu | chat_id=%s", chat_id)
            return None

    def _send_message(self, chat_id: int, text: str) -> None:
        if self._application is None or self._loop is None:
            LOGGER.info("Telegram chưa sẵn sàng để gửi tin nhắn: %s", text)
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._application.bot.send_message(chat_id=chat_id, text=text),
                self._loop,
            )
            future.result(timeout=30)
            LOGGER.info("Đã gửi tin nhắn Telegram | chat_id=%s", chat_id)
        except FutureTimeoutError:
            LOGGER.warning(
                "Telegram phản hồi chậm khi gửi tin nhắn | chat_id=%s | text=%s",
                chat_id,
                text,
            )
        except Exception:
            LOGGER.exception("Không gửi được tin nhắn Telegram | chat_id=%s", chat_id)

    def _send_document(self, chat_id: int, file_path: Path, *, caption: str) -> bool:
        if self._application is None or self._loop is None:
            LOGGER.info("Telegram chưa sẵn sàng để gửi file: %s", file_path)
            return False
        try:
            write_timeout = self._telegram_upload_timeout_seconds(file_path)
            future = asyncio.run_coroutine_threadsafe(
                self._application.bot.send_document(
                    chat_id=chat_id,
                    document=file_path,
                    filename=file_path.name,
                    caption=caption,
                    write_timeout=write_timeout,
                    read_timeout=60.0,
                ),
                self._loop,
            )
            future.result(timeout=write_timeout + 60.0)
            return True
        except FutureTimeoutError:
            LOGGER.warning(
                "Telegram phản hồi chậm khi gửi video | chat_id=%s | file=%s",
                chat_id,
                file_path,
            )
            return False
        except Exception as exc:
            if exc.__class__.__name__ in {"TimedOut", "WriteTimeout"}:
                LOGGER.warning(
                    "Telegram upload video bị timeout | chat_id=%s | file=%s | error=%s",
                    chat_id,
                    file_path,
                    exc,
                )
                return False
            LOGGER.exception("Không gửi được video Telegram | chat_id=%s | file=%s", chat_id, file_path)
            return False

    def send_storage_document(self, chat_id: int, file_path: Path, *, caption: str) -> bool:
        return self._send_document(chat_id, file_path, caption=caption)

    def update_storage_status(self, job_id: str, text: str) -> bool:
        job = self.queue_manager.load_job(job_id)
        if job is None:
            return False
        return self._edit_status_message_sync(job, text)

    def _stage_text(
        self,
        job: VideoJob,
        stage: str,
        *,
        output_path: Path | None = None,
        error: str | None = None,
    ) -> str:
        formatter = _STAGE_TEXTS.get(stage)
        if formatter is None:
            return error or ""
        return formatter(job, output_path, error)

    async def _edit_message_text(self, chat_id: int, message_id: int, text: str) -> bool:
        if self._application is None:
            return False
        try:
            await self._application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
            )
            return True
        except Exception as exc:
            name = exc.__class__.__name__
            message = str(exc).lower()
            if "not modified" in message or name == "MessageNotModified":
                return True
            if name == "RetryAfter" and hasattr(exc, "retry_after"):
                retry_after = float(getattr(exc, "retry_after", 0.0) or 0.0)
                LOGGER.warning(
                    "Telegram rate limit khi edit | chat_id=%s | message_id=%s | retry_after=%.1f",
                    chat_id,
                    message_id,
                    retry_after,
                )
                await asyncio.sleep(min(retry_after, 3.0))
                try:
                    await self._application.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=message_id,
                        text=text,
                    )
                    return True
                except Exception:
                    LOGGER.exception("Không edit được message Telegram sau retry | chat_id=%s | message_id=%s", chat_id, message_id)
                    return False
            if "can't be edited" in message or "cannot be edited" in message or "not found" in message:
                LOGGER.warning(
                    "Không thể edit message Telegram | chat_id=%s | message_id=%s | error=%s",
                    chat_id,
                    message_id,
                    exc,
                )
                return False
            LOGGER.exception("Không edit được message Telegram | chat_id=%s | message_id=%s", chat_id, message_id)
            return False

    async def update_job_status_message(self, job: VideoJob, text: str) -> bool:
        if not self.telegram_config.send_progress_messages or not self.telegram_config.edit_progress_message:
            return False
        current = self.queue_manager.load_job(job.job_id) or job
        chat_id = current.telegram_chat_id or current.chat_id
        message_id = current.telegram_status_message_id
        if chat_id is None or message_id is None:
            return False
        if current.telegram_status_text == text:
            return True
        runtime_context = build_job_runtime_context(
            job_id=current.job_id,
            source=current.source.value,
            input_path=Path(current.input_path or "input"),
            output_path=Path(current.output_path) if current.output_path else None,
            worker_slot=None,
            worker_total=None,
        )
        with job_context_scope(runtime_context):
            ok = await self._edit_message_text(chat_id=chat_id, message_id=message_id, text=text)
        if ok:
            self.queue_manager.update_job(job.job_id, telegram_status_text=text, telegram_chat_id=chat_id)
        return ok

    def _edit_status_message_sync(self, job: VideoJob, text: str) -> bool:
        if self._loop is None or self._application is None:
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(self.update_job_status_message(job, text), self._loop)
            return bool(future.result(timeout=45))
        except FutureTimeoutError:
            LOGGER.warning("Telegram edit message timeout | job_id=%s | text=%s", job.job_id, text)
            return False
        except Exception:
            LOGGER.exception("Không edit được status message | job_id=%s", job.job_id)
            return False

    def _telegram_upload_timeout_seconds(self, file_path: Path) -> float:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        estimated = math.ceil(max(120.0, size_mb * 8.0))
        return float(min(900.0, estimated))

    def _effective_telegram_upload_limit_mb(self) -> int:
        configured_limit = int(self.telegram_config.max_video_send_mb)
        return min(configured_limit, self.TELEGRAM_OFFICIAL_UPLOAD_LIMIT_MB)

    def _telegram_api_settings(self) -> dict[str, Any]:
        base_url = self.telegram_config.api_base_url.strip()
        base_file_url = self.telegram_config.api_file_url.strip()
        local_mode = bool(self.telegram_config.local_mode)

        if base_url and not base_file_url and "/bot" in base_url:
            base_file_url = base_url.replace("/bot", "/file/bot", 1)
        if base_file_url and not base_url and "/file/bot" in base_file_url:
            base_url = base_file_url.replace("/file/bot", "/bot", 1)

        return {
            "base_url": base_url,
            "base_file_url": base_file_url,
            "local_mode": local_mode,
        }

    def _is_chat_allowed(self, chat_id: int) -> bool:
        allowed_ids = [int(item) for item in self.telegram_config.allowed_chat_ids]
        if not allowed_ids:
            return bool(self.telegram_config.allow_all_chats_if_empty)
        return chat_id in allowed_ids


def _safe_telegram_error(error: str, limit: int = 180) -> str:
    cleaned = " ".join(str(error).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def json_dumps(data: Any) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)
