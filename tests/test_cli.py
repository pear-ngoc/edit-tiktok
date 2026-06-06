from unittest.mock import patch

import cli
import app
from models import AppConfig


def test_no_arg_dispatches_to_processing() -> None:
    with patch("app.process_default") as mocked:
        cli.main([])
    mocked.assert_called_once()


def test_cli_lut_override() -> None:
    captured: dict[str, object] = {}

    def fake_process_default(project_root, overrides=None, **kwargs):
        captured["overrides"] = overrides

    with patch("app.process_default", side_effect=fake_process_default):
        cli.main(["process", "--lut", "example.cube", "--lut", "second.cube"])

    assert captured["overrides"]["lut"] == ["example.cube", "second.cube"]
    assert "no_lut" not in captured["overrides"] or not captured["overrides"]["no_lut"]


def test_cli_no_lut_override() -> None:
    captured: dict[str, object] = {}

    def fake_process_default(project_root, overrides=None, **kwargs):
        captured["overrides"] = overrides

    with patch("app.process_default", side_effect=fake_process_default):
        cli.main(["process", "--no-lut"])

    assert captured["overrides"]["no_lut"] is True


def test_cli_subtitle_overrides() -> None:
    captured: dict[str, object] = {}

    def fake_process_default(project_root, overrides=None, **kwargs):
        captured["overrides"] = overrides

    with patch("app.process_default", side_effect=fake_process_default):
        cli.main(
            [
                "process",
                "--subtitles",
                "--burn-captions",
                "--subtitle-language",
                "vi",
                "--whisper-model",
                "medium",
            ]
        )

    assert captured["overrides"]["subtitles"] is True
    assert captured["overrides"]["burn_captions"] is True
    assert captured["overrides"]["subtitle_language"] == "vi"
    assert captured["overrides"]["whisper_model"] == "medium"


def test_cli_no_subtitles_override() -> None:
    captured: dict[str, object] = {}

    def fake_process_default(project_root, overrides=None, **kwargs):
        captured["overrides"] = overrides

    with patch("app.process_default", side_effect=fake_process_default):
        cli.main(["process", "--no-subtitles", "--no-burn-captions"])

    assert captured["overrides"]["no_subtitles"] is True
    assert captured["overrides"]["burn_captions"] is False


def test_cli_caption_overrides() -> None:
    captured: dict[str, object] = {}

    def fake_process_default(project_root, overrides=None, **kwargs):
        captured["overrides"] = overrides

    with patch("app.process_default", side_effect=fake_process_default):
        cli.main(
            [
                "process",
                "--caption-max-chars-per-line",
                "18",
                "--caption-max-lines",
                "2",
                "--caption-max-words",
                "6",
                "--caption-max-duration",
                "2.4",
                "--caption-position",
                "bottom",
                "--caption-font-size",
                "60",
            ]
        )

    assert captured["overrides"]["caption_max_chars_per_line"] == 18
    assert captured["overrides"]["caption_max_lines"] == 2
    assert captured["overrides"]["caption_max_words"] == 6
    assert captured["overrides"]["caption_max_duration"] == 2.4
    assert captured["overrides"]["caption_position"] == "bottom"
    assert captured["overrides"]["caption_font_size"] == 60


def test_cli_preflight_dispatch() -> None:
    with patch("app.preflight_only") as mocked:
        cli.main(["preflight"])
    mocked.assert_called_once()


def test_cli_watch_dispatch() -> None:
    with patch("app.watch") as mocked:
        cli.main(["watch"])
    mocked.assert_called_once()


def test_cli_telegram_dispatch() -> None:
    with patch("app.telegram") as mocked:
        cli.main(["telegram"])
    mocked.assert_called_once()


def test_cli_worker_dispatch() -> None:
    with patch("app.worker") as mocked:
        cli.main(["worker", "--telegram", "--watch-input"])
    mocked.assert_called_once()
    kwargs = mocked.call_args.kwargs
    assert kwargs["enable_telegram"] is True
    assert kwargs["watch_input"] is True


def test_cli_clear_dispatch() -> None:
    with patch("app.clear_workspace") as mocked:
        cli.main(["clear", "input", "--yes"])
    mocked.assert_called_once()
    kwargs = mocked.call_args.kwargs
    assert kwargs["scope"] == "input"
    assert kwargs["yes"] is True


def test_cli_config_profile_global_override() -> None:
    captured: dict[str, object] = {}

    def fake_process_default(project_root, overrides=None, **kwargs):
        captured["kwargs"] = kwargs

    with patch("app.process_default", side_effect=fake_process_default):
        cli.main(["--config-profile", "vertical_blur"])

    assert captured["kwargs"]["config_profile"] == "vertical_blur"


def test_subtitle_wizard_yes_and_no(monkeypatch) -> None:
    config = AppConfig()
    monkeypatch.setattr("builtins.input", lambda prompt="": "y" if "Burn captions" in prompt else "vi")
    app._configure_subtitles_interactively(config)
    assert config.subtitles.enabled is True
    assert config.subtitles.burn_in is True
    assert config.subtitles.language == "vi"
    assert config.subtitles.burn_language == "vi"

    config = AppConfig()
    calls = iter(["n"])

    def no_prompt(prompt=""):
        return next(calls)

    monkeypatch.setattr("builtins.input", no_prompt)
    app._configure_subtitles_interactively(config)
    assert config.subtitles.burn_in is False
