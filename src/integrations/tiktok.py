from __future__ import annotations

import re

_TIKTOK_URL_RE = re.compile(r"https?://[^\s<>'\"()]+tiktok\.com[^\s<>'\"()]*", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,;:!?)[]}"
_LANGUAGE_CODE_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z]{2,4})?$")


def extract_tiktok_urls(text: str) -> list[str]:
    if not text:
        return []

    seen: set[str] = set()
    results: list[str] = []
    for match in _TIKTOK_URL_RE.finditer(text):
        url = match.group(0).strip().rstrip(_TRAILING_PUNCTUATION)
        if not url or "tiktok.com" not in url.lower():
            continue
        normalized = url
        if normalized in seen:
            continue
        seen.add(normalized)
        results.append(normalized)
    return results


def parse_tiktok_url_input(raw: str) -> tuple[str, str | None]:
    candidate = (raw or "").strip()
    if not candidate:
        return "", None
    if "|" not in candidate:
        return candidate, None

    url_part, suffix = candidate.rsplit("|", 1)
    normalized_url = url_part.strip().rstrip(_TRAILING_PUNCTUATION)
    language = _normalize_language_code(suffix)
    if not normalized_url:
        return candidate, None
    return normalized_url, language


def _normalize_language_code(raw: str) -> str | None:
    value = (raw or "").strip()
    if not value:
        return None
    if not _LANGUAGE_CODE_RE.fullmatch(value):
        return None
    return value.replace("_", "-").lower()
