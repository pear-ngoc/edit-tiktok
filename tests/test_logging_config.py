from __future__ import annotations

import logging

from config import default_config
from logging_config import ConsoleNoiseFilter, configure_logging
from utils.runtime_logging import NormalizedMessageFilter, RedactingFormatter


def test_configure_logging_sets_third_party_levels(tmp_path) -> None:  # noqa: ANN001
    config = default_config()
    configure_logging(tmp_path / "logs", config=config)

    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("telegram").level >= logging.WARNING
    assert logging.getLogger("faster_whisper").level >= logging.WARNING


def test_redacting_formatter_hides_telegram_token_and_api_key() -> None:
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="https://api.telegram.org/bot123456:SECRET/getUpdates x-api-key: abc123",
        args=(),
        exc_info=None,
    )

    text = formatter.format(record)

    assert "SECRET" not in text
    assert "abc123" not in text
    assert "***REDACTED***" in text


def test_normalized_message_filter_strips_duplicate_prefixes() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="[JOB job_1][LOAD_CONFIG] START hello",
        args=(),
        exc_info=None,
    )

    normalized = NormalizedMessageFilter()
    normalized.filter(record)

    assert record.msg == "START hello"


def test_console_noise_filter_suppresses_segment_noise() -> None:
    config = default_config()
    filter_ = ConsoleNoiseFilter(config)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Done in 0.12s",
        args=(),
        exc_info=None,
    )
    setattr(record, "stage", "SEGMENT 1/19")
    assert filter_.filter(record) is False


def test_console_noise_filter_rate_limits_repeated_messages() -> None:
    config = default_config()
    config.logging.suppress_repeated_messages_seconds = 60
    filter_ = ConsoleNoiseFilter(config)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Watcher heartbeat",
        args=(),
        exc_info=None,
    )

    assert filter_.filter(record) is True
    assert filter_.filter(record) is False
