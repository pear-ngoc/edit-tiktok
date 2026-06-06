from pathlib import Path

from config import default_config
from integrations.telegram_bot import TelegramBotService


def test_telegram_bot_api_settings_helper_derives_file_url(tmp_path: Path) -> None:
    config = default_config()
    config.telegram.api_base_url = "http://telegram-bot-api:8081/bot"
    config.telegram.api_file_url = ""
    config.telegram.local_mode = True
    service = TelegramBotService(tmp_path, config, queue_manager=object())

    settings = service._telegram_api_settings()

    assert settings["base_url"] == "http://telegram-bot-api:8081/bot"
    assert settings["base_file_url"] == "http://telegram-bot-api:8081/file/bot"
    assert settings["local_mode"] is True
