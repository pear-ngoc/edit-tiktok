from __future__ import annotations

from pathlib import Path

from storage.base import StorageContext, StorageUploadResult


class LocalStorageProvider:
    name = "local"

    def upload(self, file_path: Path, context: StorageContext) -> StorageUploadResult:
        size = file_path.stat().st_size if file_path.exists() else 0
        return StorageUploadResult(
            provider=self.name,
            success=True,
            local_path=file_path,
            uploaded_bytes=size,
        )
