from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

from models import StorageGoogleDriveConfig
from storage.base import PermanentStorageError, StorageContext, StorageUploadResult

LOGGER = logging.getLogger(__name__)
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


class GoogleDriveStorageProvider:
    name = "google_drive"

    def upload(self, file_path: Path, context: StorageContext) -> StorageUploadResult:
        cfg = context.config.storage.google_drive
        auth_error = _validate_auth_files(context.project_root, cfg)
        if auth_error:
            return StorageUploadResult(
                provider=self.name,
                success=False,
                local_path=file_path,
                error=auth_error,
                permanent=True,
            )
        if not cfg.folder_id.strip():
            return StorageUploadResult(
                provider=self.name,
                success=False,
                local_path=file_path,
                error="Thiếu storage.google_drive.folder_id.",
                permanent=True,
            )

        try:
            service = build_drive_service(context.project_root, cfg)
            _verify_folder_access(service, cfg.folder_id.strip(), cfg.shared_drive_id.strip())
            file_id, links = _upload_resumable(service, file_path, context)
            public_error = None
            if cfg.make_public:
                try:
                    _make_public(service, file_id, cfg.shared_drive_id.strip())
                    links = _fetch_links(service, file_id, cfg.shared_drive_id.strip())
                except Exception as exc:
                    public_error = f"Upload thành công nhưng tạo public link thất bại: {exc}"
                    LOGGER.warning("STORAGE | %s | google_drive | public link failed", context.job.job_id)
            return StorageUploadResult(
                provider=self.name,
                success=True,
                local_path=file_path,
                remote_id=file_id,
                remote_url=links.get("webViewLink") or links.get("webContentLink"),
                error=public_error,
                uploaded_bytes=file_path.stat().st_size,
            )
        except PermanentStorageError as exc:
            return StorageUploadResult(
                provider=self.name,
                success=False,
                local_path=file_path,
                error=str(exc),
                permanent=True,
            )
        except Exception as exc:
            return StorageUploadResult(
                provider=self.name,
                success=False,
                local_path=file_path,
                error=str(exc),
            )


def _resolve_credentials_path(project_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def _resolve_oauth_client_path(project_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def _resolve_oauth_token_path(project_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else project_root / path


def build_drive_service(project_root: Path, cfg: StorageGoogleDriveConfig) -> Any:
    method = (cfg.auth_method or "service_account").strip().lower()
    if method == "oauth":
        token_file = _resolve_oauth_token_path(project_root, cfg.oauth_token_file)
        credentials = _load_oauth_credentials(token_file)
        return _build_google_service(credentials)
    if method == "service_account":
        credentials_file = _resolve_credentials_path(project_root, cfg.credentials_file)
        credentials = _load_service_account_credentials(credentials_file)
        return _build_google_service(credentials)
    raise PermanentStorageError("storage.google_drive.auth_method phải là service_account hoặc oauth.")


def authorize_oauth(project_root: Path, cfg: StorageGoogleDriveConfig) -> Path:
    client_file = _resolve_oauth_client_path(project_root, cfg.oauth_client_secrets_file)
    token_file = _resolve_oauth_token_path(project_root, cfg.oauth_token_file)
    if not client_file.exists():
        raise PermanentStorageError(f"Không tìm thấy OAuth client secrets: {client_file.name}")
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except Exception as exc:
        raise PermanentStorageError("Thiếu google-auth-oauthlib để chạy OAuth.") from exc

    flow = InstalledAppFlow.from_client_secrets_file(str(client_file), DRIVE_SCOPES)
    credentials = flow.run_local_server(port=0)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json(), encoding="utf-8")
    return token_file


def _validate_auth_files(project_root: Path, cfg: StorageGoogleDriveConfig) -> str | None:
    method = (cfg.auth_method or "service_account").strip().lower()
    if method == "oauth":
        client_file = _resolve_oauth_client_path(project_root, cfg.oauth_client_secrets_file)
        token_file = _resolve_oauth_token_path(project_root, cfg.oauth_token_file)
        if not client_file.exists():
            return f"Không tìm thấy OAuth client secrets: {client_file.name}"
        if not token_file.exists():
            return "Chưa có OAuth token. Chạy: python main.py storage auth"
        return None
    if method == "service_account":
        credentials_file = _resolve_credentials_path(project_root, cfg.credentials_file)
        if not credentials_file.exists():
            return f"Không tìm thấy Google Drive credentials: {credentials_file.name}"
        return None
    return "storage.google_drive.auth_method phải là service_account hoặc oauth."


def _load_service_account_credentials(credentials_file: Path) -> Any:
    try:
        from google.oauth2 import service_account
    except Exception as exc:
        raise PermanentStorageError("Thiếu Google API dependencies.") from exc

    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_file),
            scopes=DRIVE_SCOPES,
        )
    except Exception as exc:
        raise PermanentStorageError("Google Drive credentials không hợp lệ hoặc không đọc được.") from exc
    return credentials


def _load_oauth_credentials(token_file: Path) -> Any:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except Exception as exc:
        raise PermanentStorageError("Thiếu Google OAuth dependencies.") from exc
    try:
        credentials = Credentials.from_authorized_user_file(str(token_file), DRIVE_SCOPES)
    except Exception as exc:
        raise PermanentStorageError("OAuth token không hợp lệ. Hãy chạy lại: python main.py storage auth") from exc
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            token_file.write_text(credentials.to_json(), encoding="utf-8")
        except Exception as exc:
            raise PermanentStorageError("OAuth token hết hạn và refresh thất bại. Hãy chạy lại storage auth.") from exc
    if not credentials or not credentials.valid:
        raise PermanentStorageError("OAuth token không còn hiệu lực. Hãy chạy lại: python main.py storage auth")
    return credentials


def _build_google_service(credentials: Any) -> Any:
    try:
        from googleapiclient.discovery import build
    except Exception as exc:
        raise PermanentStorageError("Thiếu google-api-python-client.") from exc
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _verify_folder_access(service: Any, folder_id: str, shared_drive_id: str) -> None:
    try:
        request = service.files().get(
            fileId=folder_id,
            fields="id,name,mimeType",
            supportsAllDrives=True,
        )
        folder = request.execute()
    except Exception as exc:
        raise PermanentStorageError("Google Drive folder inaccessible. Hãy share folder cho service account.") from exc
    if folder.get("mimeType") != "application/vnd.google-apps.folder":
        raise PermanentStorageError("storage.google_drive.folder_id không phải Google Drive folder ID.")


def _upload_resumable(service: Any, file_path: Path, context: StorageContext) -> tuple[str, dict[str, str]]:
    from googleapiclient.http import MediaFileUpload

    cfg = context.config.storage.google_drive
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    if file_path.suffix.lower() == ".mp4":
        mime_type = "video/mp4"
    metadata: dict[str, Any] = {
        "name": _remote_name(file_path, context),
        "parents": [cfg.folder_id.strip()],
        "mimeType": mime_type,
        "appProperties": {
            "job_id": context.job.job_id,
            "source": _source_value(context.job.source),
            "input_identity": (context.job.identity or "")[:64],
        },
    }
    chunk_size = max(1, int(cfg.chunk_size_mb)) * 1024 * 1024
    media = MediaFileUpload(str(file_path), mimetype=mime_type, chunksize=chunk_size, resumable=True)
    request = service.files().create(
        body=metadata,
        media_body=media,
        fields="id,webViewLink,webContentLink",
        supportsAllDrives=True,
    )
    LOGGER.info("STORAGE | %s | google_drive | uploading", context.job.job_id)
    last_logged = -10
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status is not None:
            percent = int(status.progress() * 100)
            if percent >= last_logged + 10:
                last_logged = percent
                LOGGER.info("STORAGE google_drive | %s | %s%%", context.job.job_id, percent)
    file_id = str(response["id"])
    LOGGER.info("STORAGE google_drive | %s | completed", context.job.job_id)
    return file_id, dict(response)


def _remote_name(file_path: Path, context: StorageContext) -> str:
    cfg = context.config.storage.google_drive
    if cfg.overwrite_existing:
        return file_path.name
    return f"{context.job.job_id}_{file_path.name}"


def _make_public(service: Any, file_id: str, shared_drive_id: str) -> None:
    service.permissions().create(
        fileId=file_id,
        body={"role": "reader", "type": "anyone"},
        fields="id",
        supportsAllDrives=True,
    ).execute()


def _fetch_links(service: Any, file_id: str, shared_drive_id: str) -> dict[str, str]:
    return service.files().get(
        fileId=file_id,
        fields="id,webViewLink,webContentLink",
        supportsAllDrives=True,
    ).execute()


def _source_value(source: Any) -> str:
    return str(getattr(source, "value", source))
