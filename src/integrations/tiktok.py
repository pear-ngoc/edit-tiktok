from __future__ import annotations

import re

_TIKTOK_URL_RE = re.compile(r"https?://[^\s<>'\"()]+tiktok\.com[^\s<>'\"()]*", re.IGNORECASE)
_TRAILING_PUNCTUATION = ".,;:!?)[]}"


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
