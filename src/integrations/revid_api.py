from __future__ import annotations

import json
import logging
import urllib.request
from contextlib import suppress
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

LOGGER = logging.getLogger(__name__)


def fetch_tiktok_download_info(
    tiktok_url: str,
    api_key: str,
    endpoint: str,
    timeout: int = 60,
) -> list[dict[str, object]]:
    api_url = f"{endpoint}?url={quote(tiktok_url, safe='')}"
    request = Request(
        api_url,
        method="POST",
        headers={
            "x-api-key": api_key,
            "User-Agent": "curl/8.1.2",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body = ""
        with suppress(Exception):
            body = exc.read().decode("utf-8", errors="replace")
        LOGGER.error("Revid API trả lỗi HTTP %s cho %s | body=%s", exc.code, tiktok_url, body[:500])
        raise RuntimeError(f"Revid API lỗi HTTP {exc.code}") from exc
    except URLError as exc:
        LOGGER.error("Không kết nối được Revid API cho %s: %s", tiktok_url, exc)
        raise RuntimeError("Không kết nối được Revid API") from exc

    payload = json.loads(raw)
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise RuntimeError("Phản hồi Revid API không hợp lệ")
    return [item for item in payload if isinstance(item, dict)]


def select_download_url(payload: list[dict[str, object]]) -> str:
    if not payload:
        raise RuntimeError("Revid API không trả dữ liệu tải xuống")
    first = payload[0]
    video_url = str(first.get("video_url") or "").strip()
    if video_url:
        return video_url
    download_direct = str(first.get("download_direct") or "").strip()
    if download_direct:
        return download_direct
    raise RuntimeError("Revid API không trả video_url hoặc download_direct")


def download_video_from_url(video_url: str, output_path: Path, timeout: int = 300) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    request = Request(video_url, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        LOGGER.exception("Tải video thất bại: %s", video_url)
        raise RuntimeError(f"Không tải được video: {exc}") from exc

    tmp_path.replace(output_path)
    LOGGER.info("Đã tải video: %s -> %s", video_url, output_path)
    return output_path


# ---------------------------------------------------------------------------
# tiktokdl fallback (https://tiktokios.id)
# ---------------------------------------------------------------------------


def fetch_tiktokdl_info(
    tiktok_url: str,
    endpoint: str,
    tkdl_nonce: str,
    timeout: int = 60,
) -> dict[str, object]:
    body = urlencode({
        "action": "tkdl_download",
        "tkdl_nonce": tkdl_nonce,
        "url": tiktok_url,
    }).encode("utf-8")

    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        body_err = ""
        with suppress(Exception):
            body_err = exc.read().decode("utf-8", errors="replace")[:500]
        LOGGER.error("tiktokdl API lỗi HTTP %s cho %s | body=%s", exc.code, tiktok_url, body_err)
        raise RuntimeError(f"tiktokdl API lỗi HTTP {exc.code}") from exc
    except URLError as exc:
        LOGGER.error("Không kết nối được tiktokdl cho %s: %s", tiktok_url, exc)
        raise RuntimeError("Không kết nối được tiktokdl") from exc

    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Phản hồi tiktokdl không hợp lệ")
    if not payload.get("success"):
        raise RuntimeError(f"tiktokdl báo thất bại: {payload}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("tiktokdl không trả data hợp lệ")
    return data


def select_tiktokdl_url(data: dict[str, object]) -> str:
    hdplay = str(data.get("hdplay") or "").strip()
    if hdplay:
        return hdplay
    play = str(data.get("play") or "").strip()
    if play:
        return play
    wmplay = str(data.get("wmplay") or "").strip()
    if wmplay:
        return wmplay
    raise RuntimeError("tiktokdl không trả hdplay / play / wmplay")
