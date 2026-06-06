from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from models import AppConfig, VideoJob


@dataclass(slots=True)
class StorageContext:
    project_root: Path
    job: VideoJob
    config: AppConfig
    is_subtitle: bool = False
    progress_callback: Callable[[str, str], None] | None = None


@dataclass(slots=True)
class StorageUploadResult:
    provider: str
    success: bool
    local_path: Path
    remote_id: str | None = None
    remote_url: str | None = None
    error: str | None = None
    uploaded_bytes: int = 0
    permanent: bool = False


class StorageProvider(Protocol):
    name: str

    def upload(self, file_path: Path, context: StorageContext) -> StorageUploadResult:
        ...


class PermanentStorageError(RuntimeError):
    pass


class TransientStorageError(RuntimeError):
    pass
