from pathlib import Path


def test_docker_files_exist() -> None:
    root = Path(__file__).resolve().parent.parent
    assert (root / "Dockerfile").exists()
    assert (root / "docker-compose.yml").exists()
    assert (root / ".dockerignore").exists()
    assert (root / ".env.example").exists()


def test_docker_compose_includes_telegram_bot_api_service() -> None:
    root = Path(__file__).resolve().parent.parent
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert "telegram-bot-api" in compose
    assert "ghcr.io/bots-house/docker-telegram-bot-api" in compose
    assert "TELEGRAM_BOT_API_BASE_URL" in compose
    assert "TELEGRAM_BOT_API_FILE_URL" in compose
    assert "./output:/app/output:ro" in compose
