from pathlib import Path

import pytest

from integrations.revid_api import (
    fetch_tiktokdl_info,
    select_tiktokdl_url,
)
from models import TiktokdlFallbackConfig


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
