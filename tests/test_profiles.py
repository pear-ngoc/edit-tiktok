from pathlib import Path

import app
from config import apply_overrides, default_config, load_config, save_config
from models import AppConfig
from utils.profiles import (
    list_saved_profiles,
    load_profile_config,
    merge_profile_config,
    sanitize_profile_name,
    save_profile_config,
)


def test_profile_name_sanitization() -> None:
    assert sanitize_profile_name("Tiktok Burn Caption VI") == "tiktok_burn_caption_vi"
    assert sanitize_profile_name("  ..My/Profile??  ") == "my_profile"


def test_saving_wizard_config_to_configs_folder(tmp_path: Path) -> None:
    project_root = tmp_path
    config = AppConfig()
    config.video.mode = "blur"

    path = save_profile_config(project_root, "vertical blur", config)

    assert path == project_root / "configs" / "vertical_blur.yaml"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "profile:" in text
    assert "created_at:" in text
    assert "Created from wizard" in text


def test_preventing_overwrite_unless_confirmed(tmp_path: Path) -> None:
    project_root = tmp_path
    config = AppConfig()
    save_profile_config(project_root, "vertical_blur", config)

    try:
        save_profile_config(project_root, "vertical_blur", config, overwrite=False)
        assert False, "Expected FileExistsError"
    except FileExistsError:
        pass


def test_listing_saved_profiles(tmp_path: Path) -> None:
    project_root = tmp_path
    config = AppConfig()
    save_profile_config(project_root, "no_lut_fast", config)
    save_profile_config(project_root, "vertical_blur", config)

    profiles = list_saved_profiles(project_root)
    assert [path.stem for path in profiles] == ["no_lut_fast", "vertical_blur"]


def test_loading_saved_profile(tmp_path: Path) -> None:
    project_root = tmp_path
    config = AppConfig()
    config.video.aspect_ratio = "1:1"
    save_profile_config(project_root, "square", config)

    profile = load_profile_config(project_root, "square")
    assert profile["profile"]["name"] == "square"
    assert profile["video"]["aspect_ratio"] == "1:1"


def test_merging_config_priority(tmp_path: Path) -> None:
    project_root = tmp_path
    base_config = default_config()
    base_config.video.aspect_ratio = "4:3"
    base_config.encoder.preset = "fast"
    save_config(project_root, base_config)

    loaded = load_config(project_root)
    assert loaded.video.aspect_ratio == "4:3"
    assert loaded.encoder.preset == "fast"

    profile_data = {
        "profile": {"name": "vertical_blur"},
        "video": {"aspect_ratio": "1:1", "mode": "blur"},
        "encoder": {"preset": "quality"},
    }
    merged = merge_profile_config(loaded, profile_data)
    assert merged.video.aspect_ratio == "1:1"
    assert merged.video.mode == "blur"
    assert merged.encoder.preset == "quality"

    overridden = apply_overrides(
        merged,
        {
            "aspect": "9:16",
            "preset": "balanced",
        },
    )
    assert overridden.video.aspect_ratio == "9:16"
    assert overridden.encoder.preset == "balanced"


def test_cli_profile_not_found_behavior(tmp_path: Path, capsys) -> None:
    app.process_default(tmp_path, config_profile="missing_profile")
    out = capsys.readouterr().out
    assert "Không tìm thấy cấu hình đã lưu" in out


def test_wizard_saves_profile_interactively(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig()
    responses = iter(["y", "My Vertical Profile"])

    monkeypatch.setattr("builtins.input", lambda prompt="": next(responses))
    app._maybe_save_wizard_profile(tmp_path, config)

    saved = tmp_path / "configs" / "my_vertical_profile.yaml"
    assert saved.exists()
