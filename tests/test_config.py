from pathlib import Path

import pytest

from config import ensure_config_file, load_config


def test_config_created_when_missing(tmp_path: Path) -> None:
    path = ensure_config_file(tmp_path)
    assert path.exists()
    config = load_config(tmp_path)
    assert config.processing.input_dir == "input"
    assert config.video.aspect_ratio == "9:16"
    assert config.subtitles.enabled is True
    assert config.subtitles.model_size == "medium"
    assert config.subtitles.output_vtt is False
    assert config.subtitles.burn_language == "auto"
    assert config.subtitles.device == "auto"
    assert config.subtitles.compute_type == "auto"
    assert config.subtitles.output_dir == "output/subtitles"
    assert config.subtitles.word_timestamps is True
    assert config.formatting.max_chars_per_line == 20
    assert config.formatting.max_lines == 2
    assert config.formatting.max_words_per_cue == 7
    assert config.formatting.caption_font_size == 58
    assert config.logging.level == "INFO"
    assert config.logging.per_job_logs is True
    assert config.logging.retain_failed_temp is True
    assert config.logging.progress_interval_seconds == 10.0


def test_environment_overrides_for_telegram_and_revid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
telegram:
  bot_token: ""
revid_api:
  api_key: ""
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "telegram-token")
    monkeypatch.setenv("REVID_API_KEY", "revid-key")

    config = load_config(tmp_path, config_path=config_path)
    assert config.telegram.bot_token == "telegram-token"
    assert config.revid_api.api_key == "revid-key"


def test_environment_overrides_for_local_telegram_bot_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
telegram:
  api_base_url: ""
  api_file_url: ""
  local_mode: false
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("TELEGRAM_BOT_API_BASE_URL", "http://telegram-bot-api:8081/bot")
    monkeypatch.setenv("TELEGRAM_BOT_API_FILE_URL", "http://telegram-bot-api:8081/file/bot")
    monkeypatch.setenv("TELEGRAM_BOT_API_LOCAL_MODE", "true")

    config = load_config(tmp_path, config_path=config_path)
    assert config.telegram.api_base_url == "http://telegram-bot-api:8081/bot"
    assert config.telegram.api_file_url == "http://telegram-bot-api:8081/file/bot"
    assert config.telegram.local_mode is True


def test_dotenv_file_is_loaded_automatically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        """
TELEGRAM_BOT_TOKEN=bot-token-from-env-file
REVID_API_KEY=revid-key-from-env-file
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("REVID_API_KEY", raising=False)

    config = load_config(tmp_path, config_path=config_path)
    assert config.telegram.bot_token == "bot-token-from-env-file"
    assert config.revid_api.api_key == "revid-key-from-env-file"
