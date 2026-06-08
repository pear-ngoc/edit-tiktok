from pathlib import Path

import pytest

import integrations.revid_api as revid_api
from integrations.revid_api import (
    fetch_tiktokdl_info,
    select_tiktokdl_url,
)
from models import TiktokdlFallbackConfig


@pytest.fixture(autouse=True)
def clear_tiktokdl_nonce_cache() -> None:
    revid_api._TIKTOKDL_NONCE_CACHE.clear()


class TestSelectTiktokdlUrl:
    def test_prefers_hdplay(self) -> None:
        data = {
            "hdplay": "https://example.com/hd.mp4",
            "play": "https://example.com/play.mp4",
            "wmplay": "https://example.com/wm.mp4",
        }
        assert select_tiktokdl_url(data) == "https://example.com/hd.mp4"

    def test_falls_back_to_play(self) -> None:
        data = {
            "play": "https://example.com/play.mp4",
            "wmplay": "https://example.com/wm.mp4",
        }
        assert select_tiktokdl_url(data) == "https://example.com/play.mp4"

    def test_falls_back_to_wmplay(self) -> None:
        data = {"wmplay": "https://example.com/wm.mp4"}
        assert select_tiktokdl_url(data) == "https://example.com/wm.mp4"

    def test_raises_when_no_url(self) -> None:
        data = {"author": "test"}
        with pytest.raises(RuntimeError, match="không trả hdplay"):
            select_tiktokdl_url(data)

    def test_strips_whitespace(self) -> None:
        data = {"hdplay": "  https://example.com/hd.mp4  ", "play": ""}
        assert select_tiktokdl_url(data) == "https://example.com/hd.mp4"

    def test_handles_empty_strings(self) -> None:
        data = {"hdplay": "", "play": "", "wmplay": ""}
        with pytest.raises(RuntimeError, match="không trả hdplay"):
            select_tiktokdl_url(data)


class TestFetchTiktokdlInfo:
    def test_returns_data_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_response = {
            "success": True,
            "data": {
                "hdplay": "https://example.com/hd.mp4",
                "author": {"unique_id": "test_user"},
            },
        }
        import json

        class MockResponse:
            def read(self):
                return json.dumps(mock_response).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            "integrations.revid_api.urlopen",
            lambda *args, **kwargs: MockResponse(),
        )

        result = fetch_tiktokdl_info(
            tiktok_url="https://vt.tiktok.com/ZSQdLSnHN/",
            endpoint="https://tiktokios.id/wp-admin/admin-ajax.php",
            tkdl_nonce="458d52a803",
        )
        assert result["hdplay"] == "https://example.com/hd.mp4"

    def test_refreshes_nonce_after_security_check_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        responses = [
            {"success": False, "data": {"message": "Security check failed."}},
            '<input type="hidden" id="tkdl_nonce" name="tkdl_nonce" value="fb75a1f5d9" />',
            {"success": True, "data": {"hdplay": "https://example.com/fresh.mp4"}},
        ]
        seen_requests: list[tuple[str, str, bytes | None]] = []

        class MockResponse:
            def __init__(self, payload: object):
                self.payload = payload

            def read(self):
                if isinstance(self.payload, str):
                    return self.payload.encode()
                return json.dumps(self.payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def fake_urlopen(request, timeout=0):  # noqa: ANN001
            seen_requests.append((request.get_method(), request.full_url, request.data))
            return MockResponse(responses.pop(0))

        monkeypatch.setattr("integrations.revid_api.urlopen", fake_urlopen)

        result = fetch_tiktokdl_info(
            tiktok_url="https://vt.tiktok.com/ZSQdLSnHN/",
            endpoint="https://tiktokios.id/wp-admin/admin-ajax.php",
            tkdl_nonce="458d52a803",
        )

        assert result["hdplay"] == "https://example.com/fresh.mp4"
        assert seen_requests[0][0] == "POST"
        assert b"tkdl_nonce=458d52a803" in (seen_requests[0][2] or b"")
        assert seen_requests[1] == ("GET", "https://tiktokios.id/", None)
        assert b"tkdl_nonce=fb75a1f5d9" in (seen_requests[2][2] or b"")

    def test_uses_cached_nonce_on_next_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        responses = [
            {"success": False, "data": {"message": "Security check failed."}},
            '<input type="hidden" id="tkdl_nonce" name="tkdl_nonce" value="fb75a1f5d9" />',
            {"success": True, "data": {"hdplay": "https://example.com/fresh.mp4"}},
            {"success": True, "data": {"hdplay": "https://example.com/cached.mp4"}},
        ]
        seen_posts: list[bytes] = []

        class MockResponse:
            def __init__(self, payload: object):
                self.payload = payload

            def read(self):
                if isinstance(self.payload, str):
                    return self.payload.encode()
                return json.dumps(self.payload).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        def fake_urlopen(request, timeout=0):  # noqa: ANN001
            if request.get_method() == "POST":
                seen_posts.append(request.data or b"")
            return MockResponse(responses.pop(0))

        monkeypatch.setattr("integrations.revid_api.urlopen", fake_urlopen)

        first = fetch_tiktokdl_info(
            tiktok_url="https://vt.tiktok.com/ZSQdLSnHN/",
            endpoint="https://tiktokios.id/wp-admin/admin-ajax.php",
            tkdl_nonce="458d52a803",
        )
        second = fetch_tiktokdl_info(
            tiktok_url="https://vt.tiktok.com/ZSQdLSnHN/",
            endpoint="https://tiktokios.id/wp-admin/admin-ajax.php",
            tkdl_nonce="458d52a803",
        )

        assert first["hdplay"] == "https://example.com/fresh.mp4"
        assert second["hdplay"] == "https://example.com/cached.mp4"
        assert b"tkdl_nonce=458d52a803" in seen_posts[0]
        assert b"tkdl_nonce=fb75a1f5d9" in seen_posts[1]
        assert b"tkdl_nonce=fb75a1f5d9" in seen_posts[2]

    def test_raises_on_success_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        class MockResponse:
            def read(self):
                return json.dumps({"success": False, "data": None}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            "integrations.revid_api.urlopen",
            lambda *args, **kwargs: MockResponse(),
        )

        with pytest.raises(RuntimeError, match="báo thất bại"):
            fetch_tiktokdl_info(
                tiktok_url="https://vt.tiktok.com/ZSQdLSnHN/",
                endpoint="https://tiktokios.id/wp-admin/admin-ajax.php",
                tkdl_nonce="458d52a803",
            )

    def test_raises_on_non_dict_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        class MockResponse:
            def read(self):
                return b"not json"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(
            "integrations.revid_api.urlopen",
            lambda *args, **kwargs: MockResponse(),
        )

        with pytest.raises(json.JSONDecodeError):
            fetch_tiktokdl_info(
                tiktok_url="https://vt.tiktok.com/ZSQdLSnHN/",
                endpoint="https://tiktokios.id/wp-admin/admin-ajax.php",
                tkdl_nonce="458d52a803",
            )


class TestTiktokdlFallbackConfig:
    def test_default_values(self) -> None:
        cfg = TiktokdlFallbackConfig()
        assert cfg.enabled is False
        assert cfg.endpoint == "https://tiktokios.id/wp-admin/admin-ajax.php"
        assert cfg.tkdl_nonce == "458d52a803"
        assert cfg.timeout_seconds == 60
        assert cfg.download_timeout_seconds == 300
