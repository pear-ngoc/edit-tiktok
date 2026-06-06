from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any

from config import config_from_dict, deep_merge, dump_simple_yaml
from models import AppConfig
from utils.files import sanitize_filename


def get_profiles_dir(project_root: Path) -> Path:
    return project_root / "configs"


def list_saved_profiles(project_root: Path) -> list[Path]:
    profiles_dir = get_profiles_dir(project_root)
    if not profiles_dir.exists():
        return []
    return sorted(
        path
        for path in profiles_dir.glob("*.yaml")
        if path.is_file() and path.name != ".gitkeep"
    )


def sanitize_profile_name(name: str) -> str:
    cleaned = sanitize_filename(name, replacement="_")
    cleaned = cleaned.rsplit(".", 1)[0]
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("._-").lower()
    return cleaned or "profile"


def save_profile_config(
    project_root: Path,
    name: str,
    config: AppConfig,
    *,
    overwrite: bool = False,
    description: str = "Created from wizard",
) -> Path:
    profiles_dir = get_profiles_dir(project_root)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_profile_name(name)
    path = profiles_dir / f"{safe_name}.yaml"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Configuration already exists: {path}")

    payload: dict[str, Any] = {
        "profile": {
            "name": safe_name,
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "description": description,
        }
    }
    payload.update(asdict(config))
    path.write_text(dump_simple_yaml(payload), encoding="utf-8")
    return path


def load_profile_config(project_root: Path, name: str) -> dict[str, Any]:
    path = profile_path(project_root, name)
    if not path.exists():
        raise FileNotFoundError(f"Saved config profile not found: {name}")
    return _load_simple_yaml(path)


def merge_profile_config(base_config: AppConfig, profile_config: dict[str, Any]) -> AppConfig:
    base_dict = asdict(base_config)
    merged = deep_merge(base_dict, {key: value for key, value in profile_config.items() if key != "profile"})
    return config_from_dict(merged)


def profile_path(project_root: Path, name: str) -> Path:
    return get_profiles_dir(project_root) / f"{sanitize_profile_name(name)}.yaml"


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except ModuleNotFoundError:
        from config import parse_simple_yaml

        return parse_simple_yaml(text)
