from __future__ import annotations

import asyncio
import logging
import math
from pathlib import Path
from typing import Protocol

from models import AppConfig, JobSource
from storage.base import PermanentStorageError, StorageContext, StorageUploadResult

LOGGER = logging.getLogger(__name__)


class TelegramSender(Protocol):
    def send_storage_document(self, chat_id: int, file_path: Path, *, caption: str) -> bool:
        ...

    def update_storage_status(self, job_id: str, text: str) -> bool:
        ...


class TelegramStorageProvider:
    name = "telegram"

    def __init__(self, sender: TelegramSender | None = None) -> None:
        self.sender = sender

    def upload(self, file_path: Path, context: StorageContext) -> StorageUploadResult:
        job = context.job
        chat_id = _target_chat_id(context)
        if chat_id is None:
            return StorageUploadResult(
                provider=self.name,
                success=False,
                local_path=file_path,
                error="Thiếu Telegram default_chat_id cho local job.",
                permanent=True,
            )
        if self.sender is None:
            return StorageUploadResult(
                provider=self.name,
                success=False,
                local_path=file_path,
                error="Telegram runtime chưa sẵn sàng để gửi output.",
                permanent=True,
            )

        max_bytes = int(context.config.storage.telegram.max_file_size_mb) * 1024 * 1024
        size = file_path.stat().st_size
        if size > max_bytes:
            text = "⚠️ Video đã render xong nhưng vượt giới hạn gửi Telegram.\nVideo đã được giữ trong hệ thống."
            self._update(context, text)
            return StorageUploadResult(
                provider=self.name,
                success=False,
                local_path=file_path,
                error=f"File vượt giới hạn Telegram {context.config.storage.telegram.max_file_size_mb} MB.",
                uploaded_bytes=0,
                permanent=True,
            )

        self._update(context, "📤 Đang tải output lên Telegram...")
        caption = _build_caption(context, file_path) if context.config.storage.telegram.send_caption else ""
        sent = self.sender.send_storage_document(chat_id, file_path, caption=caption)
        if not sent:
            return StorageUploadResult(
                provider=self.name,
                success=False,
                local_path=file_path,
                error="Không gửi được file qua Telegram.",
                uploaded_bytes=0,
            )
        LOGGER.info("STORAGE | %s | telegram | completed", job.job_id)
        return StorageUploadResult(
            provider=self.name,
            success=True,
            local_path=file_path,
            remote_id=str(chat_id),
            uploaded_bytes=size,
        )

    def _update(self, context: StorageContext, text: str) -> None:
        if self.sender is None:
            return
        try:
            self.sender.update_storage_status(context.job.job_id, text)
        except Exception:
            LOGGER.exception("Không cập nhật được Telegram storage status | job_id=%s", context.job.job_id)


def _target_chat_id(context: StorageContext) -> int | None:
    job = context.job
    if job.source == JobSource.TELEGRAM_TIKTOK and job.chat_id is not None:
        return int(job.chat_id)
    configured = context.config.storage.telegram.default_chat_id
    return int(configured) if configured is not None else None


def _build_caption(context: StorageContext, file_path: Path) -> str:
    backend = context.config.encoder.backend
    lines = [
        "✅ Render hoàn tất",
        "",
        f"File: {file_path.name}",
        f"Job: {context.job.job_id}",
        f"Backend: {backend}",
    ]
    return "\n".join(lines)


class DirectTelegramSender:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.telegram_config = config.telegram
        self._bot = None
        self._request = None

    def send_storage_document(self, chat_id: int, file_path: Path, *, caption: str) -> bool:
        try:
            return asyncio.run(self._send_document(chat_id, file_path, caption=caption))
        except Exception:
            LOGGER.exception("Không gửi được Telegram storage document | chat_id=%s | file=%s", chat_id, file_path.name)
            return False

    def update_storage_status(self, job_id: str, text: str) -> bool:
        return False

    async def _send_document(self, chat_id: int, file_path: Path, *, caption: str) -> bool:
        bot = self._build_bot()
        timeout = _telegram_upload_timeout_seconds(file_path)
        await bot.send_document(
            chat_id=chat_id,
            document=file_path,
            filename=file_path.name,
            caption=caption,
            write_timeout=timeout,
            read_timeout=60.0,
        )
        return True

    def _build_bot(self):
        if self._bot is not None:
            return self._bot
        try:
            from telegram import Bot
            from telegram.request import HTTPXRequest
        except Exception as exc:
            raise PermanentStorageError("Thiếu python-telegram-bot để gửi Telegram.") from exc
        token = self.telegram_config.bot_token.strip()
        if not token:
            raise PermanentStorageError("Thiếu Telegram bot token.")
        request = HTTPXRequest(
            connection_pool_size=4,
            read_timeout=60.0,
            write_timeout=60.0,
            connect_timeout=10.0,
            pool_timeout=5.0,
            media_write_timeout=float(self.config.storage.timeout_seconds),
        )
        kwargs = _telegram_api_kwargs(self.telegram_config)
        self._request = request
        self._bot = Bot(token=token, request=request, **kwargs)
        return self._bot


def _telegram_upload_timeout_seconds(file_path: Path) -> float:
    size_mb = file_path.stat().st_size / (1024 * 1024)
    return float(min(900.0, math.ceil(max(120.0, size_mb * 8.0))))


def _telegram_api_kwargs(telegram_config) -> dict[str, object]:  # noqa: ANN001
    base_url = telegram_config.api_base_url.strip()
    base_file_url = telegram_config.api_file_url.strip()
    if base_url and not base_file_url and "/bot" in base_url:
        base_file_url = base_url.replace("/bot", "/file/bot", 1)
    if base_file_url and not base_url and "/file/bot" in base_file_url:
        base_url = base_file_url.replace("/file/bot", "/bot", 1)
    kwargs: dict[str, object] = {"local_mode": bool(telegram_config.local_mode)}
    if base_url:
        kwargs["base_url"] = base_url
    if base_file_url:
        kwargs["base_file_url"] = base_file_url
    return kwargs
