from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is on path
_root = Path(__file__).parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import httpx


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    headers: dict | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = json_data or {
        "text": "test",
        "language": "en",
        "duration": 1.0,
        "segments": [],
        "words": [],
    }
    resp.text = json.dumps(resp.json.return_value)
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


@pytest.fixture
def groq_config() -> "GroqTranscriptionConfig":
    from models import GroqTranscriptionConfig

    return GroqTranscriptionConfig(
        enabled=True,
        api_key="sk-test-groq-key-12345",
        base_url="https://api.groq.com/openai/v1",
        model="whisper-large-v3-turbo",
        temperature=0.0,
        response_format="verbose_json",
        timestamp_granularities=["segment", "word"],
        timeout_seconds=60,
        connect_timeout_seconds=10,
        retry_attempts=2,
        retry_delay_seconds=1.0,
        fallback_to_local=True,
        max_concurrent_requests=1,
        audio_format="mp3",
        audio_sample_rate=16000,
        audio_channels=1,
        audio_bitrate="96k",
        chunking_enabled=False,
        chunk_duration_seconds=600,
        chunk_overlap_seconds=1.0,
    )


@pytest.fixture
def subtitles_config(groq_config: "GroqTranscriptionConfig") -> "SubtitlesConfig":
    from models import SubtitlesConfig

    return SubtitlesConfig(
        enabled=True,
        backend="auto",
        model_size="medium",
        language="auto",
        output_srt=True,
        output_vtt=False,
        burn_in=False,
        word_timestamps=True,
        groq=groq_config,
    )


@pytest.fixture
def no_key_groq_config() -> "GroqTranscriptionConfig":
    from models import GroqTranscriptionConfig

    return GroqTranscriptionConfig(enabled=True, api_key="")


@pytest.fixture
def no_key_subtitles_config(
    no_key_groq_config: "GroqTranscriptionConfig",
) -> "SubtitlesConfig":
    from models import SubtitlesConfig

    return SubtitlesConfig(backend="auto", groq=no_key_groq_config)


# ---------------------------------------------------------------------------
# GroqConfig dataclass
# ---------------------------------------------------------------------------


def test_groq_config_defaults() -> None:
    from models import GroqTranscriptionConfig

    cfg = GroqTranscriptionConfig()
    assert cfg.model == "whisper-large-v3-turbo"
    assert cfg.response_format == "verbose_json"
    assert "segment" in cfg.timestamp_granularities
    assert "word" in cfg.timestamp_granularities
    assert cfg.fallback_to_local is True
    assert cfg.chunking_enabled is True
    assert cfg.chunk_duration_seconds == 600


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def test_auto_selects_groq_when_key_in_config(
    subtitles_config: "SubtitlesConfig",
) -> None:
    from processing.transcription import TranscriptionManager

    manager = TranscriptionManager(subtitles_config, None)
    assert manager.resolve_backend() == "groq"


def test_auto_selects_local_when_no_key(
    no_key_subtitles_config: "SubtitlesConfig",
) -> None:
    from processing.transcription import TranscriptionManager

    manager = TranscriptionManager(no_key_subtitles_config, None)
    assert manager.resolve_backend() == "faster-whisper"


def test_auto_selects_local_when_no_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models import GroqTranscriptionConfig, SubtitlesConfig
    from processing.transcription import TranscriptionManager

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = SubtitlesConfig(backend="auto", groq=GroqTranscriptionConfig(api_key=""))
    manager = TranscriptionManager(cfg, None)
    assert manager.resolve_backend() == "faster-whisper"


def test_explicit_groq_without_key_warns(
    no_key_subtitles_config: "SubtitlesConfig",
) -> None:
    from models import SubtitlesConfig
    from processing.transcription import TranscriptionManager

    cfg = SubtitlesConfig(backend="groq", groq=no_key_subtitles_config.groq)
    manager = TranscriptionManager(cfg, None)
    assert manager.resolve_backend() == "faster-whisper"


def test_explicit_faster_whisper(subtitles_config: "SubtitlesConfig") -> None:
    from models import SubtitlesConfig
    from processing.transcription import TranscriptionManager

    cfg = SubtitlesConfig(backend="faster-whisper", groq=subtitles_config.groq)
    manager = TranscriptionManager(cfg, None)
    assert manager.resolve_backend() == "faster-whisper"


def test_env_key_enables_groq_backend(
    monkeypatch: pytest.MonkeyPatch,
    subtitles_config: "SubtitlesConfig",
) -> None:
    from models import SubtitlesConfig
    from processing.transcription import TranscriptionManager

    monkeypatch.setenv("GROQ_API_KEY", "sk-from-env")
    cfg = SubtitlesConfig(backend="auto", groq=subtitles_config.groq)
    manager = TranscriptionManager(cfg, None)
    assert manager.resolve_backend() == "groq"


# ---------------------------------------------------------------------------
# Groq HTTP request structure
# ---------------------------------------------------------------------------


def test_groq_request_uses_correct_model_and_format(
    tmp_path: Path,
    groq_config: "GroqTranscriptionConfig",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from processing.transcription.groq import _upload_and_transcribe_chunk

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    captured: dict = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            captured["url"] = url
            captured["files"] = kwargs.get("files") or {}
            return _mock_response(
                200,
                {
                    "text": "Hello world",
                    "language": "en",
                    "duration": 1.5,
                    "segments": [
                        {
                            "id": 0,
                            "text": "Hello world",
                            "start": 0.0,
                            "end": 1.5,
                            "words": [
                                {"word": "Hello", "start": 0.0, "end": 0.5},
                                {"word": "world", "start": 0.6, "end": 1.5},
                            ],
                        }
                    ],
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5},
                        {"word": "world", "start": 0.6, "end": 1.5},
                    ],
                },
            )

    monkeypatch.setattr(httpx, "Client", FakeClient)

    result = _upload_and_transcribe_chunk(
        audio_file,
        groq_config,
        language="en",
        api_key="sk-test-key",
        chunk_index=0,
        total_chunks=1,
        job_context=None,
    )

    files = captured.get("files", {})
    model_values = [v for k, v in files.items() if k == "model"]
    assert any("whisper-large-v3-turbo" in str(v) for v in model_values)
    format_values = [v for k, v in files.items() if k == "response_format"]
    assert any("verbose_json" in str(v) for v in format_values)
    assert "timestamp_granularities[]" in files
    assert len(result.segments) == 1
    assert len(result.words) == 2


def test_groq_language_auto_not_included(
    tmp_path: Path,
    groq_config: "GroqTranscriptionConfig",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from processing.transcription.groq import _upload_and_transcribe_chunk

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    captured: dict = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            captured["files"] = kwargs.get("files") or {}
            return _mock_response(200, {"text": "ok", "language": "vi", "segments": [], "words": []})

    monkeypatch.setattr(httpx, "Client", FakeClient)

    _upload_and_transcribe_chunk(
        audio_file,
        groq_config,
        language="auto",
        api_key="sk-test",
        chunk_index=0,
        total_chunks=1,
        job_context=None,
    )

    assert "language" not in captured.get("files", {})


def test_groq_explicit_language_sent(
    tmp_path: Path,
    groq_config: "GroqTranscriptionConfig",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from processing.transcription.groq import _upload_and_transcribe_chunk

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    captured: dict = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            captured["files"] = kwargs.get("files") or {}
            return _mock_response(200, {"text": "ok", "language": "vi", "segments": [], "words": []})

    monkeypatch.setattr(httpx, "Client", FakeClient)

    _upload_and_transcribe_chunk(
        audio_file,
        groq_config,
        language="vi",
        api_key="sk-test",
        chunk_index=0,
        total_chunks=1,
        job_context=None,
    )

    files = captured.get("files", {})
    assert "language" in files
    assert files["language"][1] == "vi"


# ---------------------------------------------------------------------------
# Groq response normalization
# ---------------------------------------------------------------------------


def test_groq_response_normalizes_segments_and_words() -> None:
    from processing.transcription.groq import _GroqVerboseResponse, _normalize_response

    raw = _GroqVerboseResponse(
        text="Hello world",
        language="en",
        duration=1.5,
        segments=[
            {
                "id": 0,
                "text": "Hello world",
                "start": 0.0,
                "end": 1.5,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.6, "end": 1.5},
                ],
            }
        ],
        words=[
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.6, "end": 1.5},
        ],
    )

    result = _normalize_response(raw)

    assert len(result.segments) == 1
    assert result.segments[0].text == "Hello world"
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.5
    assert len(result.segments[0].words) == 2
    assert len(result.words) == 2
    assert result.words[0].text == "Hello"
    assert result.words[1].text == "world"


def test_groq_chunk_offset_applied() -> None:
    from processing.transcription.groq import _GroqVerboseResponse, _normalize_response

    raw = _GroqVerboseResponse(
        text="test",
        language="en",
        duration=10.0,
        segments=[{"id": 0, "text": "test", "start": 0.0, "end": 10.0, "words": []}],
        words=[{"word": "test", "start": 0.0, "end": 10.0}],
    )

    result = _normalize_response(raw, chunk_offset=60.0)

    assert result.segments[0].start == 60.0
    assert result.segments[0].end == 70.0
    assert result.words[0].start == 60.0
    assert result.words[0].end == 70.0


# ---------------------------------------------------------------------------
# Overlap deduplication
# ---------------------------------------------------------------------------


def test_deduplicate_words_removes_duplicates() -> None:
    from processing.transcription.groq import _deduplicate_words
    from processing.transcription.models import TranscriptionWord

    words = [
        TranscriptionWord(text="hello", start=0.0, end=0.5),
        TranscriptionWord(text="world", start=0.6, end=1.0),
        TranscriptionWord(text="world", start=0.65, end=1.05),
        TranscriptionWord(text="again", start=2.0, end=2.5),
    ]

    result = _deduplicate_words(words)

    assert len(result) == 3
    assert result[0].text == "hello"
    assert result[1].text == "world"
    assert result[2].text == "again"


def test_deduplicate_words_keeps_adjacent_distinct_words() -> None:
    from processing.transcription.groq import _deduplicate_words
    from processing.transcription.models import TranscriptionWord

    words = [
        TranscriptionWord(text="Jeśli", start=0.02, end=0.26),
        TranscriptionWord(text="ktoś", start=0.26, end=0.56),
        TranscriptionWord(text="trzaśnie", start=0.56, end=0.98),
        TranscriptionWord(text="drzwiami", start=0.98, end=1.30),
    ]

    result = _deduplicate_words(words)

    assert [word.text for word in result] == ["Jeśli", "ktoś", "trzaśnie", "drzwiami"]


def test_groq_upload_sends_all_timestamp_granularities(
    tmp_path: Path,
    groq_config: "GroqTranscriptionConfig",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from processing.transcription.groq import _upload_and_transcribe_chunk

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    captured: dict = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            captured["files"] = kwargs.get("files") or []
            return _mock_response(
                200,
                {
                    "text": "ok",
                    "language": "pl",
                    "segments": [{"text": "ok", "start": 0.0, "end": 1.0, "words": []}],
                    "words": [{"word": "ok", "start": 0.0, "end": 1.0}],
                },
            )

    monkeypatch.setattr(httpx, "Client", FakeClient)

    _upload_and_transcribe_chunk(
        audio_file,
        groq_config,
        language="pl",
        api_key="sk-test",
        chunk_index=0,
        total_chunks=1,
        job_context=None,
    )

    files = captured.get("files", [])
    granularity_values = [value[1] for key, value in files if key == "timestamp_granularities[]"]
    assert granularity_values == ["segment", "word"]


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def test_groq_429_retry_then_success(
    tmp_path: Path,
    groq_config: "GroqTranscriptionConfig",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    sleep_times: list = []

    monkeypatch.setattr("time.sleep", lambda d: sleep_times.append(d))

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(429, {}, {"Retry-After": "2"})
            return _mock_response(
                200,
                {"text": "ok", "language": "en", "segments": [], "words": []},
            )

    monkeypatch.setattr(httpx, "Client", FakeClient)

    from processing.transcription.groq import _upload_and_transcribe_chunk

    result = _upload_and_transcribe_chunk(
        audio_file,
        groq_config,
        language="en",
        api_key="sk-test",
        chunk_index=0,
        total_chunks=1,
        job_context=None,
    )

    assert call_count == 2
    assert 2.0 in sleep_times


def test_groq_401_does_not_retry(
    tmp_path: Path,
    groq_config: "GroqTranscriptionConfig",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            nonlocal call_count
            call_count += 1
            return _mock_response(401, {}, {})

    monkeypatch.setattr(httpx, "Client", FakeClient)

    from processing.transcription.groq import GroqPermanentError, _upload_and_transcribe_chunk

    with pytest.raises(GroqPermanentError, match="401"):
        _upload_and_transcribe_chunk(
            audio_file,
            groq_config,
            language="en",
            api_key="sk-bad-key",
            chunk_index=0,
            total_chunks=1,
            job_context=None,
        )

    assert call_count == 1


def test_groq_500_retries_then_succeeds(
    tmp_path: Path,
    groq_config: "GroqTranscriptionConfig",
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    monkeypatch.setattr("time.sleep", lambda d: None)

    audio_file = tmp_path / "test.mp3"
    audio_file.write_bytes(b"fake audio")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(500, {}, {})
            return _mock_response(
                200,
                {"text": "ok", "language": "en", "segments": [], "words": []},
            )

    monkeypatch.setattr(httpx, "Client", FakeClient)

    from processing.transcription.groq import _upload_and_transcribe_chunk

    result = _upload_and_transcribe_chunk(
        audio_file,
        groq_config,
        language="en",
        api_key="sk-test",
        chunk_index=0,
        total_chunks=1,
        job_context=None,
    )

    assert call_count == 2


# ---------------------------------------------------------------------------
# Fallback to local
# ---------------------------------------------------------------------------


def test_groq_falls_back_to_local_when_enabled(
    tmp_path: Path,
    subtitles_config: "SubtitlesConfig",
) -> None:
    from models import SubtitlesConfig
    from processing.transcription import TranscriptionManager
    from processing.transcription.models import TranscriptionResult

    cfg = SubtitlesConfig(backend="groq", groq=subtitles_config.groq)
    cfg.groq.fallback_to_local = True

    local_called = False

    def fake_groq(*args, **kwargs):
        raise RuntimeError("Groq failed")

    def fake_local(*args, **kwargs):
        nonlocal local_called
        local_called = True
        return TranscriptionResult(
            backend="faster-whisper",
            text="hello",
            language="en",
            duration=1.0,
            segments=[],
            words=[],
        )

    with patch(
        "processing.transcription.manager.GroqTranscriptionBackend.transcribe",
        fake_groq,
    ):
        with patch(
            "processing.transcription.manager.LocalWhisperBackend.transcribe",
            fake_local,
        ):
            manager = TranscriptionManager(cfg, None)
            result = manager.transcribe(tmp_path / "video.mp4", "auto")

    assert local_called
    assert result.backend == "faster-whisper"


def test_groq_raises_when_fallback_disabled(
    tmp_path: Path,
    subtitles_config: "SubtitlesConfig",
) -> None:
    from models import SubtitlesConfig
    from processing.transcription import TranscriptionManager

    cfg = SubtitlesConfig(backend="groq", groq=subtitles_config.groq)
    cfg.groq.fallback_to_local = False

    def fake_groq(*args, **kwargs):
        raise RuntimeError("Groq failed")

    with patch(
        "processing.transcription.manager.GroqTranscriptionBackend.transcribe",
        fake_groq,
    ):
        manager = TranscriptionManager(cfg, None)
        with pytest.raises(RuntimeError, match="Groq failed"):
            manager.transcribe(tmp_path / "video.mp4", "auto")


# ---------------------------------------------------------------------------
# Local not called when Groq succeeds
# ---------------------------------------------------------------------------


def test_local_not_called_when_groq_succeeds(
    tmp_path: Path,
    subtitles_config: "SubtitlesConfig",
) -> None:
    from models import SubtitlesConfig
    from processing.transcription import TranscriptionManager
    from processing.transcription.models import TranscriptionResult

    cfg = SubtitlesConfig(backend="groq", groq=subtitles_config.groq)

    local_called = False

    def fake_groq(*args, **kwargs):
        return TranscriptionResult(
            backend="groq",
            text="hello",
            language="en",
            duration=1.0,
            segments=[],
            words=[],
        )

    def fake_local(*args, **kwargs):
        nonlocal local_called
        local_called = True
        return TranscriptionResult(
            backend="faster-whisper",
            text="hello",
            language="en",
            duration=1.0,
            segments=[],
            words=[],
        )

    with patch(
        "processing.transcription.manager.GroqTranscriptionBackend.transcribe",
        fake_groq,
    ):
        with patch(
            "processing.transcription.manager.LocalWhisperBackend.transcribe",
            fake_local,
        ):
            manager = TranscriptionManager(cfg, None)
            result = manager.transcribe(tmp_path / "video.mp4", "auto")

    assert not local_called
    assert result.backend == "groq"


# ---------------------------------------------------------------------------
# Cue splitting
# ---------------------------------------------------------------------------


def test_groq_words_feed_into_cue_splitter() -> None:
    from processing.subtitles import split_words_into_caption_cues, SubtitleWord

    words = [
        SubtitleWord(text="Xin", start=0.0, end=0.2),
        SubtitleWord(text="chào,", start=0.22, end=0.45),
        SubtitleWord(text="bạn", start=1.0, end=1.3),
    ]

    cues = split_words_into_caption_cues(
        words,
        max_chars_per_line=20,
        max_lines=2,
        max_chars_per_cue=40,
        max_words_per_cue=7,
        min_duration=0.7,
        max_duration=2.6,
        pause_threshold=0.45,
    )

    assert len(cues) == 2
    assert cues[0].text.startswith("Xin chào,")
    assert cues[1].text == "bạn"


def test_groq_and_local_produce_same_cues() -> None:
    from processing.subtitles import split_words_into_caption_cues, SubtitleWord

    words = [
        SubtitleWord(text="Những", start=0.0, end=0.18),
        SubtitleWord(text="anh", start=0.20, end=0.32),
        SubtitleWord(text="em", start=0.34, end=0.45),
        SubtitleWord(text="đi", start=0.48, end=0.56),
        SubtitleWord(text="trước", start=0.58, end=0.72),
        SubtitleWord(text="hoặc", start=0.75, end=0.90),
        SubtitleWord(text="trong", start=0.93, end=1.05),
        SubtitleWord(text="ngành", start=1.08, end=1.24),
        SubtitleWord(text="cho", start=1.28, end=1.40),
        SubtitleWord(text="mình", start=1.43, end=1.58),
        SubtitleWord(text="xin", start=1.61, end=1.75),
        SubtitleWord(text="lời", start=1.78, end=1.90),
        SubtitleWord(text="khuyên", start=1.93, end=2.10),
    ]

    cues = split_words_into_caption_cues(
        words,
        max_chars_per_line=20,
        max_lines=2,
        max_chars_per_cue=40,
        max_words_per_cue=7,
        min_duration=0.7,
        max_duration=2.6,
        pause_threshold=0.45,
    )

    assert len(cues) >= 2
    assert all(len(cue.lines) <= 2 for cue in cues)


# ---------------------------------------------------------------------------
# API key redaction
# ---------------------------------------------------------------------------


def test_api_key_redacted() -> None:
    from processing.transcription.groq import _redact_key

    assert "sk-test-abc" not in _redact_key("sk-test-abcdefgh")
    result = _redact_key("sk-test-abcdefgh")
    assert result.startswith("sk-") and "***" in result and result.endswith("gh")
    assert _redact_key("") == "<not set>"
    assert _redact_key("abc") == "***"


# ---------------------------------------------------------------------------
# Semaphore
# ---------------------------------------------------------------------------


def test_semaphore_same_for_same_pid() -> None:
    from processing.transcription.groq import _get_semaphore

    sem1 = _get_semaphore(2)
    sem2 = _get_semaphore(2)
    assert sem1 is sem2


# ---------------------------------------------------------------------------
# No audio video
# ---------------------------------------------------------------------------


def test_no_audio_video_returns_none() -> None:
    from ffmpeg_tools.probe import VideoInfo
    from processing.subtitles import generate_subtitles_for_video
    from models import SubtitlesConfig

    video = MagicMock(spec=Path)
    video.exists.return_value = True

    cfg = SubtitlesConfig(enabled=True, backend="auto")

    with patch("processing.subtitles.probe_video") as mock_probe:
        mock_probe.return_value = VideoInfo(
            path=video,
            duration=10.0,
            width=1080,
            height=1920,
            fps=30.0,
            has_audio=False,
        )
        result = generate_subtitles_for_video(
            video,
            cfg,
            project_root=Path("/tmp"),
            temp_root=Path("/tmp"),
            debug=False,
        )

    assert result is None
    mock_probe.assert_called_once_with(video)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


def test_cli_transcription_backend_arg() -> None:
    from cli import _process_parser

    parser = _process_parser()
    ns = parser.parse_args(["--transcription-backend", "groq"])
    assert ns.transcription_backend == "groq"

    ns2 = parser.parse_args(["--transcription-backend", "auto"])
    assert ns2.transcription_backend == "auto"


def test_cli_groq_model_arg() -> None:
    from cli import _process_parser

    parser = _process_parser()
    ns = parser.parse_args(["--groq-transcription-model", "whisper-large-v3"])
    assert ns.groq_transcription_model == "whisper-large-v3"


def test_cli_groq_fallback_group() -> None:
    from cli import _process_parser

    parser = _process_parser()
    ns1 = parser.parse_args(["--groq-fallback-local"])
    assert ns1.groq_fallback_local is True

    ns2 = parser.parse_args(["--no-groq-fallback-local"])
    assert ns2.groq_fallback_local is False


# ---------------------------------------------------------------------------
# Config overrides
# ---------------------------------------------------------------------------


def test_config_override_transcription_backend() -> None:
    from config import apply_overrides
    from models import AppConfig

    cfg = AppConfig()
    overridden = apply_overrides(cfg, {"transcription_backend": "groq"})
    assert overridden.subtitles.backend == "groq"


def test_config_override_groq_model() -> None:
    from config import apply_overrides
    from models import AppConfig

    cfg = AppConfig()
    overridden = apply_overrides(cfg, {"groq_transcription_model": "whisper-large-v3"})
    assert overridden.subtitles.groq.model == "whisper-large-v3"


def test_config_override_groq_fallback() -> None:
    from config import apply_overrides
    from models import AppConfig

    cfg = AppConfig()
    assert cfg.subtitles.groq.fallback_to_local is True
    overridden = apply_overrides(cfg, {"groq_fallback_local": False})
    assert overridden.subtitles.groq.fallback_to_local is False


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------


def test_wizard_subtitles_unchanged() -> None:
    from app import _configure_subtitles_interactively
    from models import AppConfig

    cfg = AppConfig()

    def input_side_effect(prompt):
        if "burn captions" in prompt.lower():
            return "y"
        if "language" in prompt.lower():
            return "vi"
        return ""

    with patch("builtins.input", side_effect=input_side_effect):
        _configure_subtitles_interactively(cfg)

    assert cfg.subtitles.burn_in is True
    assert cfg.subtitles.language == "vi"


def test_wizard_does_not_ask_backend() -> None:
    from app import _configure_subtitles_interactively
    import inspect

    source = inspect.getsource(_configure_subtitles_interactively)
    assert "backend" not in source.lower()
    assert "groq" not in source.lower()


# ---------------------------------------------------------------------------
# Telegram stage texts
# ---------------------------------------------------------------------------


def test_telegram_groq_stage_texts() -> None:
    from integrations.telegram_bot import _STAGE_TEXTS

    assert "generating_subtitles_groq" in _STAGE_TEXTS
    assert "fallback_to_local" in _STAGE_TEXTS
    assert "Groq" in _STAGE_TEXTS["generating_subtitles_groq"](None)


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def test_resolve_api_key_prefers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from models import GroqTranscriptionConfig
    from processing.transcription.groq import _resolve_api_key

    monkeypatch.setenv("GROQ_API_KEY", "sk-from-env")
    cfg = GroqTranscriptionConfig(api_key="sk-from-config")
    assert _resolve_api_key(cfg) == "sk-from-env"


def test_resolve_api_key_falls_back_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from models import GroqTranscriptionConfig
    from processing.transcription.groq import _resolve_api_key

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = GroqTranscriptionConfig(api_key="sk-from-config")
    assert _resolve_api_key(cfg) == "sk-from-config"


def test_resolve_language_normalizes_groq_names() -> None:
    from processing.transcription.groq import _GroqVerboseResponse, _resolve_language

    assert _resolve_language(_GroqVerboseResponse(language="English", text="", duration=0, segments=[], words=[])) == "en"
    assert _resolve_language(_GroqVerboseResponse(language="Vietnamese", text="", duration=0, segments=[], words=[])) == "vi"
    assert _resolve_language(_GroqVerboseResponse(language="Japanese", text="", duration=0, segments=[], words=[])) == "ja"
    assert _resolve_language(_GroqVerboseResponse(language="Korean", text="", duration=0, segments=[], words=[])) == "ko"
    assert _resolve_language(_GroqVerboseResponse(language="Chinese", text="", duration=0, segments=[], words=[])) == "zh"
    assert _resolve_language(_GroqVerboseResponse(language="Spanish", text="", duration=0, segments=[], words=[])) == "es"
    assert _resolve_language(_GroqVerboseResponse(language="French", text="", duration=0, segments=[], words=[])) == "fr"
    assert _resolve_language(_GroqVerboseResponse(language=None, text="", duration=0, segments=[], words=[])) is None
    # Unknown language falls through
    assert _resolve_language(_GroqVerboseResponse(language="AncientSumerian", text="", duration=0, segments=[], words=[])) == "AncientSumerian"


def test_groq_response_normalizes_without_top_level_words() -> None:
    # Simulate Groq response that has no top-level `words` (only segment-level)
    from processing.transcription.groq import _GroqVerboseResponse, _normalize_response

    raw = _GroqVerboseResponse(
        text="Don't forget to subscribe.",
        language="English",
        duration=1.968,
        segments=[
            {
                "id": 0,
                "text": " Don't forget to subscribe.",
                "start": 0,
                "end": 1.9599999,
                "words": [
                    {"word": "Don't", "start": 0.04, "end": 0.36},
                    {"word": "forget", "start": 0.36, "end": 0.7},
                    {"word": "to", "start": 0.7, "end": 1.1},
                    {"word": "subscribe.", "start": 1.1, "end": 1.52},
                ],
            }
        ],
        words=[],  # No top-level words — Groq default
    )

    result = _normalize_response(raw)

    assert len(result.segments) == 1
    assert len(result.words) == 4
    assert result.words[0].text == "Don't"
    assert result.words[0].start == 0.04
    assert result.words[-1].text == "subscribe."
    assert result.segments[0].words[0].text == "Don't"


def test_groq_handles_null_segments_and_words() -> None:
    # Groq may return null instead of [] for segments/words
    from processing.transcription.groq import _parse_verbose_response

    data = {
        "text": "Test",
        "language": "en",
        "duration": 1.0,
        "segments": None,  # null instead of []
        "words": None,  # null instead of []
    }

    parsed = _parse_verbose_response(data)
    assert parsed.segments == []
    assert parsed.words == []
    assert parsed.text == "Test"
